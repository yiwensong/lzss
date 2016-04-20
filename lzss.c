#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "lzss_help.h"
#include "common.h"

void dummy(char *buf,size_t sz)
{
  char k;
  for(int i=0;i<sz;i++)
    k = buf[i];
  return;
}

int main(int argc, char **argv)
{
  char *savename = read_string( argc, argv, "-o", NULL );
  char *compname = read_string( argc, argv, "-c", NULL );
  char *dcmpname = read_string( argc, argv, "-d", NULL );

  if( find_option( argc, argv, "-h" ) >= 0 ||
      savename == NULL ||
      (compname == NULL && dcmpname == NULL) ||
      (compname != NULL && dcmpname != NULL) )
  {
    printf( "Options:\n" );
    printf( "-h to see this help\n" );
    printf( "-c <filename> compression input file\n" );
    printf( "-d <filename> decompression input file\n" );
    printf( "-o <filename> output file\n" );
    printf( "-t print timings\n" );
    return 0;
  }

  double time;
  int t = (find_option( argc, argv, "-t" ) >= 0);

  FILE *fsave = fopen(savename,"w");
  FILE *input = compname ? fopen(compname,"r") : fopen(dcmpname,"r");

  int64_t fsize = file_size( ((compname)?compname:dcmpname) );

  unsigned char *buf = (unsigned char*) malloc(fsize * sizeof(unsigned char));

  int64_t out_size;

  if(compname)
  {
    fwrite((void*)&fsize,sizeof(int64_t),1,fsave);
    out_size = fsize;
  }
  else
  {
    fread((void*)&out_size,sizeof(int64_t),1,input);
  }
  unsigned char *out = (unsigned char*) malloc( out_size * sizeof(unsigned char) );

  int64_t cur_size = fread(buf,sizeof(unsigned char),fsize,input);
  if( compname )
  {
    out_size = compress(buf,cur_size,out);
    fprintf(stderr,"wtf dummy\n"); fflush(0);
    dummy(out,out_size);
    fprintf(stderr,"wtf human\n"); fflush(0);
    human_readable_compression(out,out_size);
    fprintf(stderr,"decompress\n"); fflush(0);
    char buf2[ 1500 ];
    decompress(out,out_size,buf2);
    fprintf(stderr,"%s\n",buf2);
  }
  else
  {
    fprintf(stderr,"wtf\n"); fflush(0);
    human_readable_compression(buf,cur_size);
    out_size = decompress(buf,cur_size,out);
  }
  fwrite(out,sizeof(unsigned char),out_size,fsave);
  
  return 0;
}
