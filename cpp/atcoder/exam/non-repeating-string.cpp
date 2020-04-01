#include <iostream>
#include <cmath>
#include <vector>

using ullong = unsigned long long;
ullong const mod = static_cast<int>(std::pow(10, 9) + 7);

ullong computeNumberOfStrings(std::size_t const    pos,
                              std::size_t const    len,
                              std::vector<int>   & counts,
                              std::string        & t)
{
	if (len <= 1) return len;

	// If we are at the first position, we can add any character.
	if (pos == 0) {
		ullong ans = 0;
		for (auto const c : counts) {
			if (counts[c - 'a']) {
				t[0] = c;
				--counts[c - 'a'];
				ans += computeNumberOfStrings(pos + 1, len, counts, t);
				++counts[c - 'a'];
			}
		}

		return ans;
	}

	// If we can add the last character.
	if (pos == len - 1) {
		for (auto const c : counts) {
			if (c != t[pos - 1]) {
				return 1;
			}
		}

		return 0;
	}

	// Recursively compute for the number of characters remaning.
	ullong ans = 0;
	for (auto const c : counts) {
		if (c != t[pos - 1] && counts[c - 'a']) {
			t[pos] = c;
			--counts[c - 'a'];
			ans += computeNumberOfStrings(pos + 1, len, counts, t);
			++counts[c - 'a'];
		}
	}

	return ans;
}

int main() {
	std::string S, T;
	std::cin >> S;

	std::vector<int> counts(26, 0);
	for (std::size_t i = 0; i < S.length(); ++i) {
		++counts[S[i] - 'a'];
	}

	std::cout << computeNumberOfStrings(0, S.length(), counts, T) << std::endl;

	return 0;
}
