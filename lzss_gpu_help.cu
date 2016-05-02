#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

#include "common.h"
#include "lzss_help.h"

#define DISP_BITS 11
#define LEN_BITS 4
#define WINDOW ((1<<DISP_BITS))
#define MAX_MATCH ((1<<LEN_BITS) + 1)
#define MIN_MATCH 2
#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) < (y)) ? (y) : (x))

uint16_t match_len(uint8_t* old, uint8_t* fwd, uint16_t fwd_max)
{
  uint16_t i;
  for(i=2;(i<min(fwd_max,MAX_MATCH)) && (old[i]==fwd[i]);i++);
  return i;
}

void window_match(uint8_t* word, uint16_t back_max, uint16_t fwd_max, match_expanded_t* dst)
{
  uint16_t best = 0;
  uint16_t offset = 0;

  for(uint16_t i=1;i<back_max;i++)
  {
    uint8_t* tmp = word - i;
    if(tmp[0] == word[0] && tmp[1] == word[1])
    {
      uint16_t curr = match_len(tmp,word,fwd_max+i);
      if(curr > best)
      {
        offset = i;
        best = curr;
      }
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

void pack_match(match_t* match, match_expanded_t* expanded)
{
  match->dl = PACK(expanded->d-1,expanded->l-2);
}

#define PUT_BIT(bit,idx) ((bit) << (7-((idx)%8)))
#define IDX_BY_BIT(arr,idx) ((arr)[(idx)/8])
#define GET_BIT(arr,idx) ((IDX_BY_BIT(arr,idx) >> (7-((idx)%8))) & 0x1)

#define MATCH_BUF_MAX (2 * WINDOW)
#define MATCH_BUF_SIZE (MATCH_BUF_MAX + MAX_MATCH)
/* Make sure flags is zeroed out before passed in */
comp_size_t compress(uint8_t* input, uint64_t input_len, uint8_t* dst, uint8_t* flags)
{
  match_expanded_t match;
  match_t m;
  uint64_t i=0;
  uint64_t w=0;
  uint64_t b=0;
  uint8_t *curr;

  for(;i<input_len;)
  {
    curr = input + i;

    uint64_t window_offset = min(i,WINDOW);
    window_match(curr, window_offset, (uint16_t) input_len-i, &match);
    if( match.l < MIN_MATCH )
    {
      /* add 0 bit and the byte */
      for(int j=0;j<max(1,match.l);j++)
      {
        IDX_BY_BIT(flags,b+j) |= PUT_BIT(0,b+j);
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
      IDX_BY_BIT(flags,b) |= PUT_BIT(1,b);
      pack_match(&m,&match);
      memcpy(dst + w,&m,sizeof(match_t));
      i += match.l;
      w += sizeof(match_t);
      b++;
    }

  }

  comp_size_t sizes;
  sizes.b = b;
  sizes.w = w;

  for(int i=0;i<(sizes.b + 7)/8;i++)
  {
    fprintf(stdout,"flag byte: %x\n",flags[i]);
  }

  fprintf(stderr,"flag bits: %ld, stuff bytes: %ld\n",b,w);
  return sizes; /* dst length can be calculated from flags and length of flags */
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

