#include <array>
#include <iostream>

int minDistance(std::string const & word1, std::string const & word2) {
	int const N = word1.length();
	int const M = word2.length();

	std::array<std::array<int, M>, N> D;

	for (int i = 0; i < N; ++i) D[i][0] = 0;
	for (int i = 0; i < M; ++i) D[0][i] = 0;

	for (int i = 1; i < N; ++i) {
		for (int j = 1; j < M; ++j) {
			if (word1[i] == word2[j]) {
				D[i][j] = D[i - 1][j - 1];
			} else {
				D[i][j] = 1 + std::min(D[i - 1][j], D[i][j - 1]);
			}
		}
	}

	return D[N - 1][M - 1];
}

int main() {
	std::string word1, word2;
	std::cin >> word1 >> word2;
	std::cout << minDistance(word1, word2) << std::endl;
	return 0;
}
