#include "common/book.h"

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char *buffer, long const size,
                             unsigned int *histo) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int const stride = blockDim.x * gridDim.x;

  while (i < size) {
      atomicAdd(&(histo[buffer[i]]), 1);
      i += stride;
  }
}

int main(void) {
  unsigned char *buffer = (unsigned char *)big_random_block(SIZE);

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  unsigned char *dev_buffer;
  unsigned int *dev_histo;

  HANDLE_ERROR(cudaMalloc((void **)&dev_buffer, SIZE));
  HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&dev_histo, 256 * sizeof(long)));
  HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));

  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int const blocks = prop.multiProcessorCount;
  histo_kernel<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);

  unsigned int histo[256];

  HANDLE_ERROR(
      cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));

  float elapsed_time;

  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("Time to generate: %3.1fms\n", elapsed_time);

  long histo_count = 0;
  for (int i = 0; i < 256; ++i) {
    histo_count += histo[i];
  }

  printf("Histogram sum: %ld\n", histo_count);

  for (int i = 0; i < SIZE; ++i) {
    --histo[buffer[i]];
  }

  for (int i = 0; i < 256; ++i) {
    if (histo[i] != 0) {
      printf("Failure at %d!\n", i);
    }
  }

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  cudaFree(dev_buffer);
  cudaFree(dev_histo);

  free(buffer);

  return 0;
}
