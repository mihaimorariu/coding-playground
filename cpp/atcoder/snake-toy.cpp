#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

int main() {
	std::uint16_t N, K;
	std::cin >> N >> K;

	std::vector<std::uint16_t> l;
	l.resize(N);
	for (std::uint16_t i = 0; i < N; ++i) {
		std::cin >> l[i];
	}

	std::sort(std::begin(l), std::end(l), std::greater<std::uint16_t>());
	std::uint32_t ans = 0;

	for (std::uint32_t i = 0; i < K; ++i) {
		ans += l[i];
	}

	std::cout << ans << std::endl;

	return 0;
}
