#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "lzss_help.h"
#include "common.h"

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


  /* Preprocessing */
  if(compname)
  {
    uint64_t fsize = file_size(compname);
    decomp_t *decomp = (decomp_t*) malloc(fsize*sizeof(uint8_t));
    size_t read = fread(decomp->content, fsize, sizeof(uint8_t),input);
    decomp->content_len = fsize;

    compressed_t *comp = lzss_compress(decomp);
    uint64_t total_len = comp->content_len + BITS_TO_CHARS(comp->flag_bits);
    size_t wrote = fwrite(comp, sizeof(uint8_t), sizeof(compressed_t) + total_len * sizeof(uint8_t), fsave);

    fprintf(stderr,"\nwrote %d bytes\n",wrote);
  }
  else
  {
    uint64_t fsize = file_size(dcmpname);
    compressed_t *comp = (compressed_t*) malloc(fsize*sizeof(uint8_t));
    size_t read = fread(comp, fsize, sizeof(uint8_t),input);

    decomp_t *decomp = lzss_decomp(comp);
    uint64_t total_len = decomp->content_len;
    size_t wrote = fwrite(decomp->content, sizeof(uint8_t), total_len * sizeof(uint8_t), fsave);

    fprintf(stderr,"input size %d bytes\n",comp->content_len);
    fprintf(stderr,"wrote %d bytes\n",wrote);
  }
  
  return 0;
}
