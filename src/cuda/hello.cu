#include <stdio.h>

const int N = 16;
const int block_size = 16;

__global__
void hello(char *a, int *b) {
  a[threadIdx.x] += b[threadIdx.x];
}

int main(int argc, char *argv[]) {
  char a[N] = "Hello ";
  int b[N] = {15, 10, 6, 0 -11, 1, 0};

  char *ad;
  int *bd;

  const int csize = N * sizeof(char);
  const int isize = N * sizeof(int);

  printf("%s", a);

  cudaMalloc((void**)&ad, csize);
  cudaMalloc((void**)&bd, isize);
  cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

  dim3 dim_block(block_size, 1);
  dim3 dim_grid(1, 1);
  hello<<<dim_grid, dim_block>>>(ad, bd);

  cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
  cudaFree(ad);;
  cudaFree(bd);

  printf("%s\n", a);

  return EXIT_SUCCESS;
}
