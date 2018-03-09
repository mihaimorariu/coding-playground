#include <iostream>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <vector>

using llong = long long;

int main() {
	std::uint64_t N;
	std::uint64_t a;

	std::cin >> N;
	std::vector<llong> S(N + 1, 0);

	for (std::uint64_t i = 1; i <= N; ++i) {
		std::cin >> a;
		S[i] = S[i - 1] + a;
	}

	llong min_diff = LLONG_MAX;
	for (std::uint64_t i = 1; i <= N - 1; ++i) {
		min_diff = std::min<llong>(min_diff, std::llabs(S[N] - 2 * S[i]));
	}

	std::cout << min_diff << std::endl;

	return 0;
}
