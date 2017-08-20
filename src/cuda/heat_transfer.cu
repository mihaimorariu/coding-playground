#include "common/book.h"
#include "common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

texture<float, 2> tex_const_src;
texture<float, 2> tex_in;
texture<float, 2> tex_out;

__global__ void copy_const_kernel(float *iptr) {
  int x      = threadIdx.x + blockIdx.x * blockDim.x;
  int y      = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float c = tex2D(tex_const_src, x, y);
  if (c != 0) { iptr[offset] = c; }
}

__global__ void blend_kernel(float *dst, bool dst_out) {
  int x      = threadIdx.x + blockIdx.x * blockDim.x;
  int y      = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float t, l, c, r, b;
  if (dst_out) {
    t = tex2D(tex_in, x, y - 1);
    l = tex2D(tex_in, x - 1, y);
    c = tex2D(tex_in, x, y);
    r = tex2D(tex_in, x + 1, y);
    b = tex2D(tex_in, x, y + 1);
  } else {
    t = tex2D(tex_out, x, y - 1);
    l = tex2D(tex_out, x - 1, y);
    c = tex2D(tex_out, x, y);
    r = tex2D(tex_out, x + 1, y);
    b = tex2D(tex_out, x, y + 1);
  }

  dst[offset] = c + SPEED * (t + b + l + r - 4 * c);
}

struct DataBlock {
  unsigned char *output_bitmap;
  float         *dev_in_src;
  float         *dev_out_src;
  float         *dev_const_src;
  CPUAnimBitmap *bitmap;
  cudaEvent_t    start, stop;
  float          total_time;
  float          frames;
};

void anim_gpu(DataBlock *d, int ticks) {
  HANDLE_ERROR(cudaEventRecord(d->start, 0));
  dim3 blocks(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  CPUAnimBitmap *bitmap = d->bitmap;

  volatile bool dst_out = true;
  for (int i = 0; i < 90; ++i) {
    float *in  = dst_out ? d->dev_in_src  : d->dev_out_src;
    float *out = dst_out ? d->dev_out_src : d->dev_in_src;

    copy_const_kernel<<<blocks, threads>>>(in);
    blend_kernel<<<blocks, threads>>>(out, dst_out);
    dst_out = !dst_out;
  }

  float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_in_src);

  HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaEventRecord(d->stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(d->stop));

  float elapsed_time;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, d->start, d->stop));

  d->total_time += elapsed_time;
  ++d->frames;
  printf("Average time per frame: %3.1f ms\n", d->total_time / d->frames);
}

void anim_exit(DataBlock *d) {
  cudaUnbindTexture(tex_in);
  cudaUnbindTexture(tex_out);
  cudaUnbindTexture(tex_const_src);

  cudaFree(d->dev_in_src);
  cudaFree(d->dev_out_src);
  cudaFree(d->dev_const_src);

  HANDLE_ERROR(cudaEventDestroy(d->start));
  HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(int argc, char *argv[]) {
  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);

  data.bitmap     = &bitmap;
  data.total_time = 0;
  data.frames     = 0;

  HANDLE_ERROR(cudaEventCreate(&data.start));
  HANDLE_ERROR(cudaEventCreate(&data.stop));

  HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));

  HANDLE_ERROR(cudaMalloc((void**)&data.dev_in_src,    bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_out_src,   bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_const_src, bitmap.image_size()));

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  HANDLE_ERROR(cudaBindTexture2D(NULL, tex_const_src, data.dev_const_src, desc, DIM, DIM, sizeof(float) * DIM));
  HANDLE_ERROR(cudaBindTexture2D(NULL, tex_in,        data.dev_in_src,    desc, DIM, DIM, sizeof(float) * DIM));
  HANDLE_ERROR(cudaBindTexture2D(NULL, tex_out,       data.dev_out_src,   desc, DIM, DIM, sizeof(float) * DIM));

  float *temp = (float*)malloc(bitmap.image_size());
  for (int i = 0; i < DIM * DIM; ++i) {
    temp[i] = 0;

    int x = i % DIM;
    int y = i / DIM;

    if (x > 300 && x < 600 && y > 310 && y < 601) { temp[i] = MAX_TEMP; }
  }

  temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
  temp[DIM * 700 + 100] = MIN_TEMP;
  temp[DIM * 300 + 300] = MIN_TEMP;
  temp[DIM * 200 + 700] = MIN_TEMP;

  for (int y = 800; y < 900; ++y) {
    for (int x = 400; x < 500; ++x) {
      temp[x + y * DIM] = MIN_TEMP;
    }
  }

  HANDLE_ERROR(cudaMemcpy(data.dev_const_src, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

  for (int y = 800; y < DIM; ++y) {
    for (int x = 0; x < 200; ++x) {
      temp[x + y * DIM] = MAX_TEMP;
    }
  }

  HANDLE_ERROR(cudaMemcpy(data.dev_in_src, temp, bitmap.image_size(), cudaMemcpyHostToDevice));

  free(temp);
  bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);

  return EXIT_SUCCESS;
}

