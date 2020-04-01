#include <cmath>
#include <iostream>
#include <map>
#include <vector>

int main() {
	int N = 0;
	std::string S1, S2;

	std::cin >> N >> S1 >> S2;

	/*
	 *    xoo  x.o  xxo
	 *    x..  x.o  ..o
	 */

	std::uint64_t count = 1;
	std::uint64_t mod   = std::pow(10, 9) + 7;

	for (int i = 0; i < N; ++i) {
		if (S1[i] == S2[i]) {
			if (i == 0) {
				count *= 3;
				count %= mod;
			} else if (S1[i - 1] == S2[i - 1]) {
				count *= 2;
				count %= mod;
			}
		} else {
			if (i == 0) {
				count *= 6;
				count %= mod;
			} else if (S1[i - 1] == S2[i - 1]) {
				count *= 2;
				count %= mod;
			} else {
				count *= 3;
				count %= mod;
			}
			++i;
		}
	}

	std::cout << count << std::endl;

	return 0;
}
