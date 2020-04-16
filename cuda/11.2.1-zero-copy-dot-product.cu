#include "common/book.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

#define imin(a, b) (a < b ? a : b)
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

int const N = 33 * 1024;
int const threads_per_block = 256;
int const blocks_per_grid =
    imin(32, (N + threads_per_block - 1) / threads_per_block);

__global__ void dot(int size, float *a, float *b, float *c) {
  __shared__ float cache[threads_per_block];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int const cache_index = threadIdx.x;

  float temp = 0;
  while (tid < size) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  cache[cache_index] = temp;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cache_index < i) {
      cache[cache_index] += cache[cache_index + i];
      __syncthreads();
    }
    i /= 2;
  }

  if (cache_index == 0) {
    c[blockIdx.x] = cache[0];
  }
}

float malloc_test(int size) {
  cudaEvent_t start, stop;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  float elapsed_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  a = (float *)malloc(size * sizeof(float));
  b = (float *)malloc(size * sizeof(float));
  partial_c = (float *)malloc(blocks_per_grid * sizeof(float));

  HANDLE_ERROR(cudaMalloc((void **)&dev_a, size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c, size * sizeof(float)));

  for (int i = 0; i < size; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  HANDLE_ERROR(cudaEventRecord(start, 0));
  HANDLE_ERROR(
      cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));
  dot<<<blocks_per_grid, threads_per_block>>>(size, dev_a, dev_b,
                                              dev_partial_c);
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
                          blocks_per_grid * sizeof(float),
                          cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  c = 0;
  for (int i = 0; i < blocks_per_grid; ++i) {
    c += partial_c[i];
  }

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_partial_c));

  free(a);
  free(b);
  free(partial_c);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Dot product: %f\n", c);

  return elapsed_time;
}

float cuda_host_malloc_test(int size) {
  cudaEvent_t start, stop;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;
  float elapsed_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaHostAlloc((void **)&a, size * sizeof(float),
                             cudaHostAllocWriteCombined | cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc((void **)&b, size * sizeof(float),
                             cudaHostAllocWriteCombined | cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc((void **)&partial_c, size * sizeof(float),
                             cudaHostAllocMapped));

  for (int i = 0; i < size; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  dot<<<blocks_per_grid, threads_per_block>>>(size, dev_a, dev_b,
                                              dev_partial_c);
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  c = 0;
  for (int i = 0; i < blocks_per_grid; ++i) {
    c += partial_c[i];
  }

  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(partial_c));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Dot product: %f\n", c);

  return elapsed_time;
}

int main(void) {
  cudaDeviceProp prop;
  int which_device;

  HANDLE_ERROR(cudaGetDevice(&which_device));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, which_device));
  if (prop.canMapHostMemory != 1) {
    printf("Device %d cannot map memory.\n", which_device);
    return 0;
  }

  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

  float elapsed_time = malloc_test(N);
  printf("Time using cudaMalloc: %.5f ms\n", elapsed_time);

  elapsed_time = cuda_host_malloc_test(N);
  printf("Time using cudaHostAlloc: %.5f ms\n", elapsed_time);

  return 0;
}
