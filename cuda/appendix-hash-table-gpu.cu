#include "common/book.h"
#include "lock.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define SIZE (100 * 1024 * 1024)
#define ELEMENTS (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES 1024

struct Entry {
  unsigned int key;
  void *value;
  Entry *next;
};

struct Table {
  size_t count;
  Entry **entries;
  Entry *pool;
};

void copy_table_to_host(Table const &table, Table &host_table) {
  host_table.count = table.count;
  host_table.entries = (Entry **)calloc(table.count, sizeof(Entry *));
  host_table.pool = (Entry *)malloc(ELEMENTS * sizeof(Entry));

  HANDLE_ERROR(cudaMemcpy(host_table.entries, table.entries,
                          table.count * sizeof(Entry *),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(host_table.pool, table.pool, ELEMENTS * sizeof(Entry),
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < table.count; i++) {
    if (host_table.entries[i] != NULL) {
      host_table.entries[i] =
          (Entry *)((size_t)host_table.entries[i] - (size_t)table.pool +
                    (size_t)host_table.pool);
    }
  }
  for (int i = 0; i < ELEMENTS; i++) {
    if (host_table.pool[i].next != NULL) {
      host_table.pool[i].next =
          (Entry *)((size_t)host_table.pool[i].next - (size_t)table.pool +
                    (size_t)host_table.pool);
    }
  }
}

void initialize_table(Table &table, int const entries, int const elements) {
  table.count = entries;
  HANDLE_ERROR(cudaMalloc((void **)&table.entries, entries * sizeof(Entry *)));
  HANDLE_ERROR(cudaMemset(table.entries, 0, entries * sizeof(Entry *)));
  HANDLE_ERROR(cudaMalloc((void **)&table.pool, elements * sizeof(Entry)));
}

void free_table(Table &table) {
  HANDLE_ERROR(cudaFree(table.entries));
  HANDLE_ERROR(cudaFree(table.pool));
}

__device__ __host__ size_t hash(unsigned int key, size_t count) {
  return key % count;
}

__global__ void add_to_table(unsigned int *keys, void **values, Table &table,
                             Lock *lock) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int const stride = blockDim.x * gridDim.x;

  while (tid < ELEMENTS) {
    unsigned int const key = keys[tid];
    size_t const hash_value = hash(key, table.count);

    for (int i = 0; i < 32; ++i) {
      if (tid % 32 == i) {
        Entry *location = &table.pool[tid];
        location->key = key;
        /*location->value = values[tid];*/
        /*lock[hash_value].lock();*/
        /*location->next = table.entries[hash_value];*/
        /*table.entries[hash_value] = location;*/
        /*lock[hash_value].unlock();*/
      }
    }

    tid += stride;
  }
}

void verify_table(Table const &dev_table) {
  Table table;
  copy_table_to_host(dev_table, table);

  int count = 0;
  for (size_t i = 0; i < table.count; ++i) {
    Entry *current = table.entries[i];
    while (current != NULL) {
      ++count;
      if (hash(current->key, table.count) != i) {
        printf("%d hashed to %ld, but located at %ld.\n", current->key,
               hash(current->key, table.count), i);
      }
      current = current->next;
    }
  }

  if (count != ELEMENTS) {
    printf("%d elements found in the hash table. Should be %ld.\n", count,
           ELEMENTS);
  } else {
    printf("All %d elements found in the hash table.\n", count);
  }

  free(table.pool);
  free(table.entries);
}

int main(void) {
  unsigned int *buffer = (unsigned int *)big_random_block(SIZE);

  /*cudaEvent_t start, stop;*/
  /*HANDLE_ERROR(cudaEventCreate(&start));*/
  /*HANDLE_ERROR(cudaEventCreate(&stop));*/
  /*HANDLE_ERROR(cudaEventRecord(start, 0));*/

  unsigned int *dev_keys;
  void **dev_values;

  HANDLE_ERROR(cudaMalloc((void **)&dev_keys, SIZE));
  HANDLE_ERROR(cudaMalloc((void **)&dev_values, SIZE));
  HANDLE_ERROR(cudaMemcpy(dev_keys, buffer, SIZE, cudaMemcpyHostToDevice));

  Table table;
  initialize_table(table, HASH_ENTRIES, ELEMENTS);

  Lock lock[HASH_ENTRIES];
  Lock *dev_lock;

  HANDLE_ERROR(cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(Lock)));
  HANDLE_ERROR(cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(Lock),
                          cudaMemcpyHostToDevice));

  add_to_table<<<60, 256>>>(dev_keys, dev_values, table, dev_lock);

  /*HANDLE_ERROR(cudaEventRecord(stop, 0));*/
  /*HANDLE_ERROR(cudaEventSynchronize(stop));*/

  /*float elapsed_time;*/
  /*HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));*/
  /*printf("Time to hash: %3.1f ms.\n", elapsed_time);*/

  /*verify_table(table);*/

  /*HANDLE_ERROR(cudaEventDestroy(start));*/
  /*HANDLE_ERROR(cudaEventDestroy(stop));*/

  free_table(table);
  cudaFree(dev_lock);
  cudaFree(dev_keys);
  cudaFree(dev_values);
  free(buffer);

  return 0;
}
