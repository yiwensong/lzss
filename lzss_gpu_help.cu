#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <cuda.h>

#include "common.h"
#include "lzss_gpu_help.h"

#define DISP_BITS 11
#define LEN_BITS 4
#define WINDOW ((1<<DISP_BITS))
#define MAX_MATCH ((1<<LEN_BITS) + 1)
#define MIN_MATCH 2
#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) < (y)) ? (y) : (x))

__device__ uint16_t match_len(uint8_t* old, uint8_t* fwd, uint16_t fwd_max)
{
  uint16_t i;
  for(i=0;(i<min(fwd_max,MAX_MATCH)) && (old[i]==fwd[i]);i++);
  return i;
}

__device__ void window_match(uint8_t* word, uint16_t back_max, uint16_t fwd_max, match_expanded_t* dst)
{
  uint16_t best = 0;
  uint16_t offset = 0;

  for(uint16_t i=1;i<back_max;i++)
  {
    uint8_t* tmp = word - i;
    uint16_t curr = match_len(tmp,word,fwd_max+i);
    if(curr > best)
    {
      offset = i;
      best = curr;
    }
  }

  dst->d = offset;
  dst->l = best;
}

#define MATCH_TOP(i) ((i) >> LEN_BITS)
#define MATCH_BOT(i) ((i) & ((1<<(LEN_BITS))-1))

void unpack_match(match_expanded_t* expanded, match_t* match)
{
  uint16_t data = match->dl;
  expanded->d = MATCH_TOP(data) + 1;
  expanded->l = MATCH_BOT(data) + 2;
}

#define TO_TOP(i) ((i) << LEN_BITS)
#define TO_BOT(i) MATCH_BOT((i))
#define PACK(d,l) (TO_TOP((d)) | TO_BOT((l)))

__device__ void pack_match(match_t* match, match_expanded_t* expanded)
{
  match->dl = PACK(expanded->d-1,expanded->l-2);
}

#define PUT_BIT(bit,idx) ((bit) << (7-((idx)%8)))
#define IDX_BY_BIT(arr,idx) ((arr)[(idx)/8])
#define GET_BIT(arr,idx) ((IDX_BY_BIT(arr,idx) >> (7-((idx)%8))) & 0x1)

/* Make sure flags is zeroed out before passed in */
__global__ void gpu_compress(uint8_t* input, uint64_t input_len, uint8_t* dst, uint8_t* flags, comp_size_t *size)
{
  /* threadIdx.x; */
  uint64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t partition_size = (global_idx * MAX_MATCH * 2 >= input_len) ?
    0 : min(input_len - global_idx * MAX_MATCH * 2, MAX_MATCH * 2);
  int64_t partition_off = (global_idx * MAX_MATCH * 2);

  input = input + partition_off;
  dst = dst + partition_off;
  flags = flags + partition_off;
  size = size + global_idx;

  match_expanded_t match;
  match_t m;
  uint64_t i=0;
  uint64_t w=0;
  uint64_t b=0;
  uint8_t *curr;
  /*
  printf("thread ID: %ld, last idx: %ld\n", global_idx + 100000, min((int64_t)(input_len-partition_off),(int64_t)partition_size));
  return;
  */
  for(;i<min((int64_t)(input_len-partition_off),(int64_t)partition_size) && partition_size > 0;)
  {
    curr = input + i;

    printf("OFFSET %5i\n",partition_off + i + 100000);

    uint64_t window_offset = min(partition_off + i,WINDOW);
    window_match(curr, window_offset, min((uint16_t) input_len-partition_off-i, (uint16_t) partition_size-i), &match);
    if( match.l < MIN_MATCH )
    {
      /* add 0 bit and the byte */
      for(int j=0;j<max(1,match.l);j++)
      {
        /* IDX_BY_BIT(flags,b+j) |= PUT_BIT(0,b+j); */
        flags[b+j] = 0;
        dst[w+j] = curr[j];
      }
      b += max(1,match.l);
      w += max(1,match.l);
      i += max(1,match.l);
    }
    else
    {
      /* match.d is displacement */
      /* match.l is length of match */
      /* IDX_BY_BIT(flags,b) |= PUT_BIT(1,b); */
      flags[b] = 1;
      pack_match(&m,&match);
      /* memcpy(dst + w,&m,sizeof(match_t)); */
      m = *((match_t*)(dst + w));
      i += match.l;
      w += sizeof(match_t);
      b++;
    }

  }

  size->b = b;
  size->w = w;
  printf("DONE %ld\n", global_idx + 1000000);
}

#define THREADS 256
#define BLOCKS ((input_len + (THREADS * MAX_MATCH * 2) + 1)/(THREADS * MAX_MATCH * 2))
#define PARTITION_SIZE ((input_len + THREADS * BLOCKS - 1)/(THREADS * BLOCKS))
comp_size_t compress(uint8_t* input, uint64_t input_len, uint8_t* dst, uint8_t* flags)
{
  uint8_t *gpu_input, *gpu_dst, *gpu_flags;
  comp_size_t *sizes, *gpu_sizes;
  printf("length: %ld, THREADS: %d, BLOCKS: %ld\n",input_len, THREADS, BLOCKS); fflush(0);
  cudaDeviceSynchronize();

  cudaMalloc(&gpu_input, input_len);
  cudaMalloc(&gpu_dst, input_len);
  cudaMalloc(&gpu_flags, input_len);
  cudaMalloc(&gpu_sizes, BLOCKS * THREADS * sizeof(comp_size_t));

  cudaMemcpy(gpu_input,input,input_len,cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  gpu_compress <<<BLOCKS,THREADS>>> (gpu_input, input_len, gpu_dst, gpu_flags, gpu_sizes);

  cudaDeviceSynchronize();

  comp_size_t cmp_size;
  cmp_size.b = 0;
  cmp_size.w = 0;

  sizes = (comp_size_t*) malloc(THREADS * BLOCKS * sizeof(comp_size_t));
  cudaMemcpy(sizes, gpu_sizes, THREADS * BLOCKS * sizeof(comp_size_t), cudaMemcpyDeviceToHost);

  uint8_t *flag_buf = (uint8_t*) malloc(input_len);
  for(int i=0;i<THREADS*BLOCKS;i++)
  {
    cudaMemcpy(dst + cmp_size.w , gpu_dst + i * PARTITION_SIZE, sizes[i].w, cudaMemcpyDeviceToHost);
    cudaMemcpy(flag_buf + cmp_size.b , gpu_flags + i * PARTITION_SIZE, sizes[i].b, cudaMemcpyDeviceToHost);
    fprintf(stdout,"bsize[%3.3d] = %3.3d, wsize[%3.3d] = %3.3d\n",i,sizes[i].b,i,sizes[i].w);
    cmp_size.b += sizes[i].b;
    cmp_size.w += sizes[i].w;
  }

  fprintf(stdout,"flag bits: %ld, stuff bytes: %ld\n",cmp_size.b,cmp_size.w);

  memset(flags,0,(cmp_size.b+7)/8);
  for(int i=0;i<cmp_size.b;i++)
  {
    IDX_BY_BIT(flags,i) |= PUT_BIT(flag_buf[i],i);
  }

  return cmp_size;
}

uint64_t decompress(uint8_t* input, uint8_t* flags, uint64_t input_len, uint8_t* dst)
{
  uint64_t b=0;
  uint64_t i=0;
  uint64_t w=0;

  register uint8_t *curr;
  register uint8_t bit;
  match_t *match;
  match_expanded_t m;

  for(;i<input_len;)
  {
    curr = input + i;
    bit = GET_BIT(flags,b);
    
    if(!bit)
    {
      dst[w] = input[i];
      w++; i++; b++;
    }
    else
    {
      match = (match_t*) curr;
      unpack_match(&m,match);
      for(int j=0;j<m.l;j++)
      {
        dst[w + j] = dst[w + j - m.d];
      }
      w += m.l;
      i += sizeof(match_t);
      b++;
    }
  }

  return w;
}


compressed_t *lzss_compress(decomp_t *decomp)
{
  uint64_t len = decomp->content_len;
  uint8_t *flag = (uint8_t*) calloc(BITS_TO_CHARS(len), sizeof(uint8_t));
  if(!flag)
  {
    return NULL;
  }
  uint8_t *buf = (uint8_t*) malloc(len * sizeof(uint8_t));
  if(!buf)
  {
    free(flag);
    return NULL;
  }

  comp_size_t comp_sizes = compress(decomp->content,len,buf,flag);
  uint64_t comp_len = comp_sizes.w;
  uint64_t flag_bits = comp_sizes.b;
  uint64_t flag_bytes = BITS_TO_CHARS(flag_bits);

  compressed_t *comp = (compressed_t*) malloc(sizeof(compressed_t) + (flag_bytes + comp_len) * sizeof(uint8_t));
  if(!comp){
    free(flag);
    free(buf);
    return NULL;
  }
  comp->file_len = len;
  comp->content_len = comp_len;
  comp->flag_bits = flag_bits;
  memcpy(comp->content,flag,flag_bytes);
  memcpy(comp->content + flag_bytes,buf,comp->content_len);

  free(flag);
  free(buf);
  return comp;
}


decomp_t *lzss_decomp(compressed_t *comp)
{
  uint64_t file_len = comp->file_len;
  uint64_t comp_len = comp->content_len;
  uint64_t flag_bits = comp->flag_bits;
  uint64_t flag_bytes = BITS_TO_CHARS(flag_bits);

  uint8_t *flag_buf = comp->content;
  uint8_t *comp_buf = comp->content + flag_bytes;

  decomp_t *decomp = (decomp_t*) malloc(sizeof(decomp_t) + (file_len)*sizeof(uint8_t));
  if(!decomp)
  {
    return NULL;
  }

  uint64_t decomp_len = decompress(comp_buf,flag_buf,comp_len,decomp->content);
  decomp->content_len = decomp_len;

  return decomp;
}

