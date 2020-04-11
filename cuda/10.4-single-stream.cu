#include "common/book.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c) {
  int const idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    int const idx1 = (idx + 1) % 256;
    int const idx2 = (idx + 2) % 256;

    float const as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
    float const bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

    c[idx] = (as + bs) / 2;
  }
}

int main(void) {
  cudaDeviceProp prop;
  int which_device;

  HANDLE_ERROR(cudaGetDevice(&which_device));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, which_device));

  if (!prop.deviceOverlap) {
    printf("Device %d will not handle overlaps, so no speed up from streams\n",
           which_device);
    return 0;
  }

  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  int *host_a, *host_b, *host_c;
  int *dev_a, *dev_b, *dev_c;

  HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

  HANDLE_ERROR(cudaHostAlloc((void **)&host_a, FULL_DATA_SIZE * sizeof(int),
                             cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&host_b, FULL_DATA_SIZE * sizeof(int),
                             cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&host_c, FULL_DATA_SIZE * sizeof(int),
                             cudaHostAllocDefault));

  for (int i = 0; i < FULL_DATA_SIZE; ++i) {
    host_a[i] = rand();
    host_b[i] = rand();
  }

  for (int i = 0; i < FULL_DATA_SIZE; i += N) {
    HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
    kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  }

  HANDLE_ERROR(cudaStreamSynchronize(stream));

  float elapsed_time;
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("Elapsed time: %3.1f ms\n", elapsed_time);

  HANDLE_ERROR(cudaFreeHost(host_a));
  HANDLE_ERROR(cudaFreeHost(host_b));
  HANDLE_ERROR(cudaFreeHost(host_c));

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  HANDLE_ERROR(cudaStreamDestroy(stream));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return 0;
}
