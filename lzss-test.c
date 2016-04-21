#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "lzss_help.h"
#include "common.h"

#define LARGE (1024 * 1024)

#define testsize 40

int main_test()
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
  uint64_t comp_size = compress((char*)text,strlen(text),buf);

  for(int i=0;i<comp_size;i++)
  {
    char_dump_bin(buf[i]);
  }
  fprintf(stderr,"\n\n\n\n\n\n\n\n");

  human_readable_compression(buf,comp_size);

  char *buf2 = (char*) malloc( strlen(text) * sizeof(char) );
  uint64_t decomp_size = decompress((char*) buf, comp_size, buf2);

  fprintf(stderr,"comp size: %d, decomp size: %d\n",(int)comp_size,(int)decomp_size);

  int cmp = memcmp(text,buf2,strlen(text) * sizeof(char));

  fprintf(stderr,"ORIGINAL:\n%s\n",text);
  fprintf(stderr,"DCOMPRSS:\n%s\n",buf2);

  fprintf(stderr,"COMPARE: %d\n",cmp);
  return 0;
}

int main(int argc, char **argv)
{

  return 0;
}
