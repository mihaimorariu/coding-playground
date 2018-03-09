#include <iostream>
#include <vector>

int main() {
	int N;
	std::vector<int> p;

	std::cin >> N;
	p.reserve(N);

	for (int i = 0; i < N; ++i) {
		std::cin >> p[i];
	}

	int count = 0;
	for (int i = 0; i < N; ++i) {
		if (p[i] == i + 1) {
			std::swap(p[i], p[i + 1]);
			++count;
		}
	}

	std::cout << count << std::endl;

	return 0;
}
