#include <iostream>
#include <cmath>
#include <vector>

int main() {
	int N;

	std::cin >> N;
	std::vector<std::uint64_t> primes(N, 0);

	for (int i = 2; i <= N; ++i) {
		int k = i;

		for (int j = 2; j <= std::sqrt(k); ++j) {
			while (k % j == 0) {
				++primes[j - 1];
				k /= j;
			}
		}

		if (k > 1) ++primes[k - 1];
	}

	std::uint64_t ans = 1;
	std::uint64_t mod = static_cast<std::uint64_t>(std::pow(10, 9) + 7);

	for (int i = 0; i < N; ++i) {
		ans = (ans * (primes[i] + 1)) % mod;
	}

	std::cout << ans << std::endl;

	return 0;
}
