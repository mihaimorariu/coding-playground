#include <stdio.h>

#define imin(a, b) (a < b ? a : b)

const int N                 = 33 * 1024;
const int threads_per_block = 256;
const int blocks_per_grid   = imin(32, (N + threads_per_block - 1 ) / threads_per_block);

__global__ void dot(float *a, float *b, float *c) {
  __shared__ float cache[threads_per_block];
  int tid         = threadIdx.x + blockIdx.x * blockDim.x;
  int cache_index = threadIdx.x;
  float temp      = 0;

  while (tid < N) {
    temp += a[tid] * b[tid];
    tid  += blockDim.x * gridDim.x;
  }

  cache[cache_index] = temp;

  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cache_index < i) {
      cache[cache_index] += cache[cache_index + i];
    }

    __syncthreads();
    i /= 2;
  }

  if (cache_index == 0) {
    c[blockIdx.x] = cache[0];
  }
}

int main(int argc, char *argv[]) {
  float *a, *b, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  a         = new float[N];
  b         = new float[N];
  partial_c = new float[blocks_per_grid];

  cudaMalloc((void**)&dev_a,         N * sizeof(float));
  cudaMalloc((void**)&dev_b,         N * sizeof(float));
  cudaMalloc((void**)&dev_partial_c, blocks_per_grid * sizeof(float));

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
  dot<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b, dev_partial_c);

  cudaMemcpy(partial_c, dev_partial_c, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);

  float c = 0;
  for (int i = 0; i < blocks_per_grid; i++) {
    c += partial_c[i];
  }

  printf("%.2f\n", c);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  delete[] a;
  delete[] b;
  delete[] partial_c;

  return EXIT_SUCCESS;
}
