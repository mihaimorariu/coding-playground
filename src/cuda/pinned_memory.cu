#include "common/book.h"

#define SIZE (10 * 1024 * 1024)

float cuda_host_alloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsed_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  for (int i = 0; i < 100; ++i) {
    if (up) {
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
    } else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsed_time;
}

float cuda_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsed_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  a = (int*)malloc(size * sizeof(a));
  HANDLE_NULL(a);
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  for (int i = 0; i < 100; ++i) {
    if (up) {
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
    } else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

  free(a);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsed_time;
}

int main(int argc, char *argv[]) {
  float elapsed_time;
  float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

  elapsed_time = cuda_malloc_test(SIZE, true);
  printf("Time using cudaMalloc: %3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy up: %3.1f\n", MB / (elapsed_time / 1000));

  elapsed_time = cuda_malloc_test(SIZE, false);
  printf("Time using cudaMalloc: %3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy down: %3.1f\n", MB / (elapsed_time / 1000));

  elapsed_time = cuda_host_alloc_test(SIZE, true);
  printf("Time using cudaHostAlloc: %3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy down: %3.1f\n", MB / (elapsed_time / 1000));

  elapsed_time = cuda_host_alloc_test(SIZE, false);
  printf("Time using cudaHostAlloc: %3.1f ms\n", elapsed_time);
  printf("\tMB/s during copy down: %3.1f\n", MB / (elapsed_time / 1000));

  return EXIT_SUCCESS;
}
