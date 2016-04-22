#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

#include "lzss_help.h"

#define WINDOW 32768
#define MIN_MATCH 4
#define min(x,y) ((x < y) ? x : y)
#define max(x,y) ((x < y) ? y : x)

uint64_t match_len(char* old, char* fwd, uint64_t fwd_max)
{
  int64_t i;
  for(i=0;i<fwd_max+1;i++)
  {
    if( old[i] != fwd[i] )
    {
      break;
    }
  }
  return (uint64_t) i;
}

void window_match(char* word, int64_t back_max, int64_t fwd_max, match_t* dst)
{
  uint16_t best = 0;
  int16_t offset = 0;

  for(int16_t i=back_max;i<0;i++)
  {
    char* tmp = word + i;
    uint16_t curr = (uint16_t) match_len(tmp,word,fwd_max);
    if(curr > best)
    {
      offset = i;
      best = curr;
    }
  }

  dst->d = (int16_t) offset;
  dst->l = (uint16_t) best;
}

#define PUT_BIT(bit,idx) ((bit) << (7-((idx)%8)))
#define IDX_BY_BIT(arr,idx) ((arr)[(idx)/8])
#define GET_BIT(arr,idx) ((IDX_BY_BIT(arr,idx) >> (7-((idx)%8))) & 0x1)

/* Make sure flags is zeroed out before passed in */
comp_size_t compress(uint8_t* input, uint64_t input_len, uint8_t* dst, uint8_t* flags)
{
  for(int i=0;i<25;i++)
  {
    fprintf(stderr,"PUT_BIT(%d): ",i);
    fprintf(stderr,"0x%lx\n",PUT_BIT(1,i));
  }
  match_t match;
  int64_t i=0;
  int64_t w=0;
  int64_t b=0;



  fprintf(stderr,"input len=%ld\n",input_len);
  for(;i<input_len;)
  {
    uint8_t *curr = input + i;
    int64_t window_offset = max(input-curr,i-WINDOW);
    window_match(curr, window_offset, min(input_len - i, 2*WINDOW), &match);

    fprintf(stderr,"\nw %ld, ",w);
    fprintf(stderr,"\nb %ld, ",b);
    
    if( match.l < MIN_MATCH )
    {
      /* add 0 bit and the byte */
      for(int j=0;j<max(1,match.l);j++)
      {
        IDX_BY_BIT(flags,b+j) |= PUT_BIT(0,b+j);
        dst[w+j] = input[i+j];
        fprintf(stderr,"0x%lx ",dst[w+j]);
      }
      fprintf(stderr," |||| ");
      for(int j=0;j<max(1,match.l);j++)
      {
        fprintf(stderr,"0x%lx ",input[i+j]);
      }
      b += max(1,match.l);
      w += max(1,match.l);
      i += max(1,match.l);
    }
    else
    {
      /* match.d is displacement */
      /* match.l is length of match */
      IDX_BY_BIT(flags,b) |= PUT_BIT(1,b);
      memcpy(dst + w,&match,sizeof(match_t));
      for(int j=0;j<4;j++)
      {
        fprintf(stderr,"0x%lx ",dst[w + j]);
      }
      i += match.l;
      w += 4;
      b++;
      fprintf(stderr,"MATCH flag is %d",GET_BIT(flags,b));
    }

    if(i>=input_len)
      fprintf(stderr,"\nloop ending i=%ld\n",i);
  }

  comp_size_t sizes;
  sizes.b = b;
  sizes.w = w;
  fprintf(stderr,"\n\n\n\n");
  return sizes; /* dst length can be calculated from flags and length of flags */
}

uint64_t decompress(uint8_t* input, uint8_t* flags, uint64_t input_len, uint8_t* dst)
{
  int64_t b=0;
  int64_t i=0;
  int64_t w=0;

  register uint8_t *curr;
  register uint8_t bit;
  match_t match;
  int16_t offset;
  uint16_t matchlen;

  for(;i<input_len;)
  {
    fprintf(stderr,"%d",i);
    fprintf(stderr,"/%d",input_len);
    fprintf(stderr," -- %d: ",b);
    curr = input + i;
    bit = GET_BIT(flags,b);
    
    if(!bit)
    {
      dst[w] = input[i];
      fprintf(stderr,"got 0x%lx, ",dst[w]);
      fprintf(stderr,"expected 0x%lx\n",input[i]);
      w++; i++; b++;
    }
    else
    {
      fprintf(stderr,"found match, ");
      // match = (match_t*) curr;
      // matchlen = match->l;
      // offset = match->d;
      memcpy(&match,curr,sizeof(match_t));
      matchlen = match.l;
      offset = match.d;
      for(int j=0;j<4;j++)
      {
        fprintf(stderr,"0x%lx ",curr[j]);
      }
      if(offset>=0)
      {
        fprintf(stderr,"\nwtf\n");
        return 0;
      }
      for(int j=0;j<matchlen;j++)
      {
        // fprintf(stderr,"offset %d, ",offset);
        dst[w + j] = dst[w + j + offset];
      }
      fprintf(stderr,"offset %d, ",offset);
      fprintf(stderr,"length %d\n",matchlen);
      w += matchlen;
      i += 4;
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
  write(2,buf,len);
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

  // fprintf(stderr,"uncompressed file length: %d\n",file_len);
  decomp_t *decomp = (decomp_t*) malloc(sizeof(decomp_t) + (file_len)*sizeof(uint8_t));
  if(!decomp)
  {
    return NULL;
  }

  uint64_t decomp_len = decompress(comp_buf,flag_buf,comp_len,decomp->content);
  fprintf(stderr,"ok\n");
  decomp->content_len = decomp_len;
  // fprintf(stderr,"file length after decomp: %d\n",decomp->content_len);

  return decomp;
}








/*

    compressed_t *comp_obj = (compressed_t*) malloc(
        sizeof(compressed_t) +
        (file_size) * sizeof(uint8_t) +
        BITS_TO_CHARS(file_size) * sizeof(uint8_t) ); 
        */


