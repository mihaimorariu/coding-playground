#include <iostream>
#include <vector>

int main() {
	int Q;
	std::cin >> Q;

	std::vector<int> C(Q);
	for (int i = 0; i < Q; ++i) {
		std::cin >> C[i];
	}

	for (int i = 0; i < Q; ++i) {
		std::vector<bool> visited(101, false);

		int k = 0;
		for (int j = i; j >= 0; --j) {
			if (!visited[C[j]]) {
				std::cout << C[j] << " ";
				if (++k == 5) break;
			}
			visited[C[j]] = true;
		}

		std::cout << std::endl;
	}

	return 0;
}
