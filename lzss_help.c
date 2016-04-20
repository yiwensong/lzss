#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "lzss_help.h"

#define WINDOW 32768
#define MIN_MATCH 4
#define min(x,y) ((x < y) ? x : y)
#define max(x,y) ((x < y) ? y : x)

size_t match_len(char* old, char* fwd, size_t fwd_max)
{
  off_t i;
  for(i=0;i<fwd_max+1;i++)
  {
    if( old[i] != fwd[i] )
    {
      break;
    }
  }
  return (size_t) i;
}

/* TODO: Replace this with the Z-algorithm later */
void window_match(char* word, off_t back_max, off_t fwd_max, match_t* dst)
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

/* Pass in last pointer WITH INFORMATION IN IT */
/* buf_to_add should be memory aligned. If not, it should be zero appended */
void write_unaligned(unsigned char* last, size_t buf_bit_size, unsigned char* buf_to_add, size_t add_size)
{
  char l = last[0];
  size_t filled = buf_bit_size % 8;
  size_t space = 8 - filled;

  char *buf = last;
  for(int i=0;i<add_size;i++)
  {
    buf[i] = (char) 0;
    buf[i] = buf[i] | l;
    buf[i] = buf[i] | (buf_to_add[i] >> filled);
    l = buf_to_add[i] << space;
  }
  buf[add_size] = l;
}

size_t compress(char* input, size_t input_len, char* dst)
{
  match_t match;
  off_t i=0;
  off_t w=0;
  for(;i<input_len;)
  {
    char *curr = input + i;
    off_t window_offset = max(input-curr,i-WINDOW);
    window_match(curr, window_offset, min(input_len - i, 2*WINDOW), &match);
    
    if( match.l < MIN_MATCH )
    {
      /* add 0 bit and the byte */
      for(int j=0;j<max(1,match.d);j++)
      {
        w++;
        char* last = dst + w/8;
        write_unaligned( (unsigned char*) last, w, (unsigned char*) curr, 1 );
        w += 8;
        i++;
      }
    }
    else
    {
      /* add 1 bit, d, and l */
      uint16_t buf[2];
      // buf[0] = 1 << 15 | match.l;
      buf[0] = match.d;
      buf[1] = match.l;

      /* write to dst + w */
      // memcpy(dst + i, buf, 2 * sizeof(uint16_t));
      char* last = dst + w/8;
      write_unaligned( (unsigned char*) last, w, (unsigned char*) buf, 4 );
      w += 32;
      i += match.l;
    }
  }

  return (w+8)/8;
}

size_t decompress(char* input, size_t input_len, char* dst)
{
  off_t i = 0;
  off_t b = 0;
  char flag;
  char tmp;
  unsigned char buf[4];
  uint16_t *uintbuf = (uint16_t*) buf;

  int16_t d;
  uint16_t l;

  for(; b / 8 < input_len ;)
  {
    flag = input[b/8] >> (7-b%8);
    if( !flag )
    {
      b++;
      /* write the next character to dst */
      dst[i] = ((unsigned char*) input)[ b/8 ] <<  (b%8);
      dst[i] |= ((unsigned char*) input)[ b/8 + 1] >> (8-b%8);
      b += 8;
      i++;
    }
    else
    {
      for(int t=0;t<4;t++)
      {
        buf[t] = ((unsigned char) input[ b/8 + t ]) << (b%8);
        buf[t] |= ((unsigned char) input[ b/8 + t + 1 ]) >> (8-(b%8));
      }

      d = (int16_t) uintbuf[0];
      l = uintbuf[1];

      fprintf(stderr,"d: %d, l: %d\n",d,l);
      fprintf(stderr,"d: %d, l: %d\n",(int16_t) uintbuf[0], uintbuf[1]);

      for(int j=0;j<l;j++)
      {
        dst[i + j] = dst[i + j + d];
      }

      i += l;
      b += 32;
    }
  }

  return (size_t) i;
}
