#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lzss_help.h"

#define LARGE (1024 * 1024)

void human_readable_compression(unsigned char *comp, size_t len)
{

  char p_buf[ LARGE ];

  off_t i = 0;
  off_t b = 0;
  char flag;
  char tmp;
  char buf[4];
  uint16_t *uintbuf = (uint16_t*) buf;

  for(; b / 8 < len ;)
  {
    flag = comp[b/8] >> (7-b%8);
    fprintf(stderr,"FLAG 0x%x b %d\n",flag,(int)b);
    if( !flag )
    {
      b++;
      /* write the next character to dst */
      p_buf[i] = comp[ b/8 ] <<  b%8;
      p_buf[i] |= comp[ b/8 + 1] >> (8-b%8);
      b += 8;
      i++;
    }
    else
    {
      for(int t=0;t<4;t++)
      {
        buf[t] = comp[ b/8 + t ] << (b%8);
        buf[t] |= comp[ b/8 + t + 1 ] >> (8-(b%8));
      }
      p_buf[i] = '(';
      i++;
      int len = sprintf(p_buf + i,"%d",(int16_t) uintbuf[0]);
      i += len;
      p_buf[i] = ',';
      i++;
      len = sprintf(p_buf + i,"%d",uintbuf[1]);
      i += len;
      p_buf[i] = ')';
      i++;
      b += 32;
    }
  }

  p_buf[i] = '\0';

  fprintf(stderr,"HUMAN READABLE COMPRESSED:\n%s\n",p_buf);

}

void char_dump_bin(unsigned char c)
{
  unsigned char buf[9];
  for(int i=0;i<8;i++)
  {
    buf[i] = '0' + ((c >> (7-i)) & 0x1);
  }
  buf[8] = '\0';
  fprintf(stderr,"%s ",buf);
}

#define testsize 40

int main()
{


  char cc = '\xff';
  unsigned char uu = '\xff';

  char_dump_bin(cc); fprintf(stderr,"\n");
  char_dump_bin(cc >> 1);fprintf(stderr,"\n");
  char_dump_bin(uu);fprintf(stderr,"\n");
  char_dump_bin(uu >> 1);fprintf(stderr,"\n\n\n\n\n\n\n\n");







  char *text = (char*) malloc( testsize * sizeof(char) );
  for(int i=0;i<testsize;i++)
  {
    text[i] = (i<testsize/2) ? 'c' : 'c';
  }
  text[testsize-1] = '\0';

  char *buf = (char*) malloc( strlen(text) * sizeof(char) );
  size_t comp_size = compress((char*)text,strlen(text),buf);

  for(int i=0;i<comp_size;i++)
  {
    char_dump_bin(buf[i]);
  }
  fprintf(stderr,"\n\n\n\n\n\n\n\n");

  human_readable_compression(buf,comp_size);

  char *buf2 = (char*) malloc( strlen(text) * sizeof(char) );
  size_t decomp_size = decompress((char*) buf, comp_size, buf2);

  fprintf(stderr,"comp size: %d, decomp size: %d\n",(int)comp_size,(int)decomp_size);

  int cmp = memcmp(text,buf2,strlen(text) * sizeof(char));

  fprintf(stderr,"ORIGINAL:\n%s\n",text);
  fprintf(stderr,"DCOMPRSS:\n%s\n",buf2);

  fprintf(stderr,"COMPARE: %d\n",cmp);
  return 0;
}
