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

struct DataStruct {
  int device_id;
  int size;
  int offset;
  float *a;
  float *b;
  float return_value;
};

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

void *routine(void *p_void_data) {
  DataStruct *data = (DataStruct *)p_void_data;
  if (data->device_id != 0) {
    HANDLE_ERROR(cudaSetDevice(data->device_id));
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  }

  int const size = data->size;
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  a = data->a;
  b = data->b;
  partial_c = (float *)malloc(blocks_per_grid * sizeof(float));

  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocks_per_grid * sizeof(float)));

  dev_a += data->offset;
  dev_b += data->offset;

  dot<<<blocks_per_grid, threads_per_block>>>(size, dev_a, dev_b,
                                              dev_partial_c);

  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
                          blocks_per_grid * sizeof(float),
                          cudaMemcpyDeviceToHost));

  c = 0;
  for (int i = 0; i < blocks_per_grid; ++i) {
    c += partial_c[i];
  }

  HANDLE_ERROR(cudaFree(partial_c));

  free(partial_c);
  data->return_value = c;
}

int main(void) {
  int device_count;

  HANDLE_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    printf("We need at least two compute 1.0 or greater devices, but only "
           "found %d\n",
           device_count);
    return 0;
  }

  cudaDeviceProp prop;
  for (int i = 0; i < 2; ++i) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    if (prop.canMapHostMemory != 1) {
      printf("Device %d cannot map memory.\n", i);
      return 0;
    }
  }

  float *a, *b;
  HANDLE_ERROR(cudaSetDevice(0));
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
  HANDLE_ERROR(cudaHostAlloc((void **)&a, N * sizeof(float),
                             cudaHostAllocWriteCombined |
                                 cudaHostAllocPortable | cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc((void **)&b, N * sizeof(float),
                             cudaHostAllocWriteCombined |
                                 cudaHostAllocPortable | cudaHostAllocMapped));

  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  DataStruct data[2];

  data[0].device_id = 0;
  data[0].offset = 0;
  data[0].size = N / 2;
  data[0].a = a;
  data[0].b = b;

  data[1].device_id = 1;
  data[0].offset = N / 2;
  data[1].size = N / 2;
  data[1].a = a + N / 2;
  data[1].b = b + N / 2;

  CUTThread thread = start_thread(routine, &data[0]);
  routine(&data[1]);
  end_thread(thread);

  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));

  printf("Value calculated: %f\n", data[0].return_value + data[1].return_value);

  return 0;
}
