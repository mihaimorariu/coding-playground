#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>
#include <vector>

std::uint64_t gcd(std::uint64_t a, std::uint64_t b) {
	if (a == 0) return b;
	if (b == 0) return a;
	return gcd(b, a % b);
}


int main() {
	int N;
	std::cin >> N;

	std::vector<std::uint64_t> T;
	T.reserve(N);

	for (int i = 0; i < N; ++i) {
		std::cin >> T[i];
	}

	std::sort(std::begin(T), std::end(T), std::greater<std::uint64_t>());
	std::uint64_t ans = T[0];

	for (int i = 1; i < N; ++i) {
		ans = ans * (T[i] / gcd(T[i], ans));
	}

	std::cout << ans << std::endl;

	return 0;
}
