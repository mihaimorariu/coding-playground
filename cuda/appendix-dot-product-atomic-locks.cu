#include "common/book.h"
#include "lock.h"

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

__global__ void dot(Lock lock, float *a, float *b, float *c) {
  __shared__ float cache[threads_per_block];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int const cache_index = threadIdx.x;

  float temp = 0;
  while (tid < N) {
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
    lock.lock();
    *c += cache[0];
    lock.unlock();
  }
}

int main(void) {
  float *a, *b, c = 0;
  float *dev_a, *dev_b, *dev_c;

  a = (float *)malloc(N * sizeof(float));
  b = (float *)malloc(N * sizeof(float));

  HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(float)));

  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_c, &c, sizeof(float), cudaMemcpyHostToDevice));

  Lock lock;
  dot<<<blocks_per_grid, threads_per_block>>>(lock, dev_a, dev_b, dev_c);

  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost));

  printf("real dot product = %.6g\n", 2 * sum_squares((float)(N - 1)));
  printf("computed dot product = %.6g\n", c);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  delete[] a;
  delete[] b;

  return 0;
}
