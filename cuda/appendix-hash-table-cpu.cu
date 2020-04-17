#include "common/book.h"

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
  Entry *first_free;
};

void initialize_table(Table &table, int const entries, int const elements) {
  table.count = entries;
  table.entries = (Entry **)calloc(entries, sizeof(Entry *));
  table.pool = (Entry *)malloc(elements * sizeof(Entry));
  table.first_free = table.pool;
}

void free_table(Table &table) {
  free(table.entries);
  free(table.pool);
}

size_t hash(unsigned int key, size_t count) { return key % count; }

void add_to_table(Table &table, unsigned int key, void *value) {
  size_t hash_value = hash(key, table.count);

  Entry *location = table.first_free++;
  location->key = key;
  location->value = value;

  location->next = table.entries[hash_value];
  table.entries[hash_value] = location;
}

void verify_table(Table const &table) {
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
}

int main(void) {
  unsigned int *buffer = (unsigned int *)big_random_block(SIZE);

  clock_t start, stop;
  start = clock();

  Table table;
  initialize_table(table, HASH_ENTRIES, ELEMENTS);

  for (unsigned int i = 0; i < ELEMENTS; ++i) {
    add_to_table(table, buffer[i], (void *)NULL);
  }

  stop = clock();
  float const elapsed_time =
      (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.f;
  printf("Time to hash: %3.1f ms.\n", elapsed_time);

  verify_table(table);
  free_table(table);
  free(buffer);

  return 0;
}
