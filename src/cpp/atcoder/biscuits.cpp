#include <iostream>

int main() {
	int N, P;
	std::vector<int> A;

	std::cin >> N >> P;
	A.reserve(N);

	for (int i = 0; i < N; ++i) {
		std::cin >> A[i];
	}

	return 0;
}
