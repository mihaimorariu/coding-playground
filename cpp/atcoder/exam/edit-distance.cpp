#include <iostream>
#include <cmath>

int isEditDistanceMaximumOne(std::string const & s1, std::string const & s2) {
	std::size_t m     = s1.length();
	std::size_t n     = s2.length();
	std::size_t i     = 0;
	std::size_t j     = 0;
	std::size_t count = 0;

	while (i < m && j < n) {
		if (s1[i] != s2[j]) {
			if (m > n) {
				++i;
			} else if (m < n) {
				++j;
			} else {
				++i;
				++j;
			}
			++count;
		} else {
			++i;
			++j;
		}
	}

	if (i < m || j < n) ++count;

	return count <= 1;
}

int main() {
	std::string S, T;
	std::cin >> S >> T;
	std::cout << (isEditDistanceMaximumOne(S, T) ? "YES" : "NO") << std::endl;

	return 0;
}
