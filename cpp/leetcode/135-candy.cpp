#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
	std::vector<int> ratings;
	int N;

	std::cin >> N;
	for (int i = 0; i < N; ++i) {
		std::cin >> ratings[i];
	}

	std::vector<int> indices;
	indices.resize(ratings.size(), 0);
	std::iota(std::begin(indices), std::end(indices), 0);
	std::sort(std::begin(indices), std::end(indices), [&ratings](int i, int j) { return ratings[i] < ratings[j]; });

	int candies       = 1;
	int total_candies = candies;
	for (std::size_t i = 1; i < indices.size(); ++i) {
		if (ratings[indices[i]] != ratings[indices[i - 1]]) {
			++candies;
		}
		total_candies += candies;
	}

	std::cout << total_candies << std::endl;

	return 0;
}
