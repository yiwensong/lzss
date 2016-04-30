#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "lzss_gpu_help.h"

__global__ void hello_world(uint64_t *wr_loc)
{
  printf("hello world from <<<%d/%d,%d/%d>>>\n",blockIdx.x,gridDim.x,threadIdx.x,blockDim.x);
  wr_loc[blockIdx.x * 256 + threadIdx.x] = blockIdx.x * 256 + threadIdx.x;
}

int main()
{
  /* dim3 blks(8,8,4);  */
  /* dim3 shape(2,2,4); */
  int blks = 10;
  int shape = 256;
  cudaDeviceSynchronize();

  uint64_t *ptr;
  cudaError_t malloc_err = cudaMalloc((void**)&ptr, 10 * 256 * sizeof(uint64_t));
  if(malloc_err != cudaSuccess)
  {
    fprintf(stderr,"YOU MALLOCED UP %s\n",cudaGetErrorString(malloc_err));
    exit(-1);
  }

  cudaDeviceSynchronize();

  hello_world <<<blks,shape>>> (ptr);

  cudaDeviceSynchronize();

  uint64_t *buf = (uint64_t*) malloc(blks*shape*sizeof(uint64_t));
  cudaError_t cpy_err = cudaMemcpy(buf,ptr,blks*shape*sizeof(uint64_t),cudaMemcpyDeviceToHost);
  if(cpy_err != cudaSuccess)
  {
    fprintf(stderr,"YOU CPYED UP %s\n",cudaGetErrorString(malloc_err));
    exit(-1);
  }

  for(uint64_t i=0;i<blks*shape;i++)
  {
    printf("%d: %d\n",i,buf[i]);
  }

  printf("ok this works\n");

  return 0;
}
