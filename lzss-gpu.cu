#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "lzss_gpu_help.h"

__global__ void hello_world()
{
  printf("hello world from <<<%d|%d,%d|%d,%d|%d>>>\n",blockIdx.x,blockDim.x,blockIdx.y,blockDim.y,blockIdx.z,blockDim.z);
}

int main()
{
  dim3 blks(10,10,10);
  dim3 shape(10,10,10);
  hello_world <<<blks,shape>>> ();
  return 0;
}
