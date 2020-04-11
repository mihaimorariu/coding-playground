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

  cudaStream_t stream0, stream1;
  HANDLE_ERROR(cudaStreamCreate(&stream0));
  HANDLE_ERROR(cudaStreamCreate(&stream1));

  int *host_a, *host_b, *host_c;
  int *dev_a0, *dev_b0, *dev_c0;
  int *dev_a1, *dev_b1, *dev_c1;

  HANDLE_ERROR(cudaMalloc((void **)&dev_a0, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b0, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c0, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_a1, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b1, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c1, N * sizeof(int)));

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

  for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
    HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream0));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream0));
    kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
    HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream0));
    HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a + i, N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream1));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b + i, N * sizeof(int),
                                 cudaMemcpyHostToDevice, stream1));
    kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
    HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c1, N * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream1));
  }

  HANDLE_ERROR(cudaStreamSynchronize(stream0));
  HANDLE_ERROR(cudaStreamSynchronize(stream1));

  float elapsed_time;
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  printf("Elapsed time: %3.1f ms\n", elapsed_time);

  HANDLE_ERROR(cudaFreeHost(host_a));
  HANDLE_ERROR(cudaFreeHost(host_b));
  HANDLE_ERROR(cudaFreeHost(host_c));

  HANDLE_ERROR(cudaFree(dev_a0));
  HANDLE_ERROR(cudaFree(dev_b0));
  HANDLE_ERROR(cudaFree(dev_c0));
  HANDLE_ERROR(cudaFree(dev_a1));
  HANDLE_ERROR(cudaFree(dev_b1));
  HANDLE_ERROR(cudaFree(dev_c1));

  HANDLE_ERROR(cudaStreamDestroy(stream0));
  HANDLE_ERROR(cudaStreamDestroy(stream1));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return 0;
}
