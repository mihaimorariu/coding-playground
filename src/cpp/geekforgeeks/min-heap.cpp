#include <assert.h>
#include <iostream>
#include <limits>

class MinHeap {
public:
	MinHeap(int capacity);

	void minHeapify(int);

	int parent(int i) const { return (i - 1) / 2; }
	int left(int i) const   { return (2 * i + 1); }
	int right(int i) const  { return (2 * i + 2); }
	int min() const         { return harr[0];     }

	int extractMin();
	void decreaseKey(int i, int new_val);

	void deleteKey(int i);
	void insertKey(int k);

	friend std::ostream &operator<<(std::ostream &os, MinHeap const & h) {
		for (int i = 0; i < h.heap_size; ++i) os << h.harr[i] << " ";
		return os;
	}

private:
	int *harr;
	int capacity;
	int heap_size;
};



MinHeap::MinHeap(int cap) {
	heap_size = 0;
	capacity  = cap;
	harr      = new int[cap];
}

void MinHeap::insertKey(int k) {
	if (heap_size == capacity) {
		throw std::overflow_error("Could not insert key, maximum capacity exceeded.");
	}

	++heap_size;
	int i = heap_size - 1;
	harr[i] = k;

	while (i != 0 && harr[parent(i)] > harr[i]) {
		std::swap(harr[i], harr[parent(i)]);
		i = parent(i);
	}
}

void MinHeap::decreaseKey(int i, int new_val) {
	assert(i >= 0 && i < heap_size && new_val < harr[i]);
	harr[i] = new_val;

	while (i != 0 && harr[parent(i)] > harr[i]) {
		std::swap(harr[parent(i)], harr[i]);
		i = parent(i);
	}
}

int MinHeap::extractMin () {
	if (heap_size <= 0) {
		return std::numeric_limits<int>::max();
	}

	if (heap_size == 1) {
		--heap_size;
		return harr[0];
	}

	int root = harr[0];
	harr[0]  = harr[heap_size - 1];

	--heap_size;
	minHeapify(0);
	
	return root;
}

void MinHeap::deleteKey(int i) {
	assert(i >= 0 && i < heap_size);
	decreaseKey(i, std::numeric_limits<int>::min());
	extractMin();
}

void MinHeap::minHeapify(int i) {
	assert(i >= 0 && i < heap_size);

	int l = left(i);
	int r = right(i);
	int smallest = i;

	if (l < heap_size && harr[l] < harr[smallest]) {
		smallest = l;
	}

	if (r < heap_size && harr[r] < harr[smallest]) {
		smallest = r;
	}

	if (smallest != i) {
		std::swap(harr[i], harr[smallest]);
		minHeapify(smallest);
	}
}


int main() {
	MinHeap h(11);

	h.insertKey(3);
	h.insertKey(2);
	h.deleteKey(1);
	h.insertKey(15);
	h.insertKey(5);
	h.insertKey(4);
	h.insertKey(45);

	std::cout << h.extractMin() << std::endl;
	std::cout << h.min() << std::endl;

	h.decreaseKey(2, 1);
	std::cout << h.min() << std::endl;

	return 0;
}
