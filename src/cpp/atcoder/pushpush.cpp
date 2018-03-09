#include <iostream>
#include <vector>

int main() {
	std::uint32_t N;
	std::cin >> N;
	std::uint64_t a;
	std::vector<std::uint64_t> b(N + 1);

	std::uint32_t j = (N + 1) / 2;
	for (std::uint32_t i = 0; i < N; ++i) {
		std::cin >> a;

		j += (i % 2 == 0 ? -i : i);
		b[j] = a;
	}

	if (N - j < j) {
		for (std::uint32_t i = j; i >= 1; --i) {
			std::cout << b[i] << " ";
		}
	} else {
		for (std::uint32_t i = j; i <= N; ++i) {
			std::cout << b[i] << " ";
		}
	}

	std::cout << std::endl;

	return 0;
}
