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

  FILE *input = compname ? fopen(compname,"r") : fopen(dcmpname,"r");
  if(!input)
  {
    return 0;
  }
  FILE *fsave = fopen(savename,"w");


  if(t)
  {
    time = read_timer();
  }

  uint64_t fsize;
  /* Preprocessing */
  if(compname)
  {
    fsize = file_size(compname);
    decomp_t *decomp = (decomp_t*) malloc(fsize*sizeof(uint8_t));
    size_t read = fread(decomp->content, fsize, sizeof(uint8_t),input);
    decomp->content_len = fsize;
    if(t)
    {
      fprintf(stdout,"Read time: %lf\n",(read_timer()-time));
    }

    compressed_t *comp = lzss_compress(decomp);
    uint64_t total_len = comp->content_len + BITS_TO_CHARS(comp->flag_bits);
    size_t wrote = fwrite(comp, sizeof(uint8_t), sizeof(compressed_t) + total_len * sizeof(uint8_t), fsave);

    if(t)
    {
      fprintf(stdout,"Mode: compress\n");
      fprintf(stdout,"Compression ratio: %lf\n", (1.0*fsize)/(1.0*wrote + 1.0e-7));
      fprintf(stdout,"Deflation: %.0lf%%\n", (1.0*wrote)/(1.0*fsize + 1.0e-7)*100);
    }
  }
  else
  {
    fsize = file_size(dcmpname);
    compressed_t *comp = (compressed_t*) malloc(fsize*sizeof(uint8_t));
    size_t read = fread(comp, fsize, sizeof(uint8_t),input);

    decomp_t *decomp = lzss_decomp(comp);
    uint64_t total_len = decomp->content_len;
    size_t wrote = fwrite(decomp->content, sizeof(uint8_t), total_len * sizeof(uint8_t), fsave);

    if(t) fprintf(stdout,"Mode: decompress\n");
  }

  if(t)
  {
    time = read_timer() - time;
    fprintf(stdout,"Input Size: %ld\n",(long int) fsize);
    fprintf(stdout,"Time elapsed: %lf\n\n\n",time);
  }
  
  return 0;
}
