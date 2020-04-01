#include "common/book.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

int main(void) {
  cudaDeviceProp prop;
  int dev;

  HANDLE_ERROR(cudaGetDevice(&dev));

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 3;

  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

  printf("ID of CUDA device closest to revision 1.3: %d\n", dev);

  HANDLE_ERROR(cudaSetDevice(dev));

  return 0;
}
