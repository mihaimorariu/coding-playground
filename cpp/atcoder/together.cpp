#include <iostream>
#include <map>
#include <vector>

int main() {
	std::map<int, int> counts;

	int N;
	std::vector<int> a;

	std::cin >> N;
	a.reserve(N);

	for (int i = 0; i < N; ++i) {
		std::cin >> a[i];

		if (!counts.count(a[i] - 1)) counts.insert({a[i] - 1, 0});
		if (!counts.count(a[i]))     counts.insert({a[i], 0});
		if (!counts.count(a[i] + 1)) counts.insert({a[i] + 1, 0});

		++counts[a[i] - 1];
		++counts[a[i]];
		++counts[a[i] + 1];
	}

	int max_count = 0, X = -1, max_i = 0;

	for (auto const & c : counts) {
		if (max_count < c.second) {
			max_count = c.second;
			X         = c.first;
		}
	}

	for (int i = 0; i < N; ++i) {
		max_i += abs(a[i] - X) <= 1;
	}

	std::cout << max_i << std::endl;

	return 0;
}
