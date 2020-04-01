#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <vector>

using umap  = std::map<uint64_t, std::uint64_t>;
using upair = std::pair<uint64_t, std::uint64_t>;

std::vector<uint64_t> dijkstra(std::uint64_t K, std::vector<umap> const & lengths) {
	std::vector<std::uint64_t> cost;
	cost.resize(lengths.size(), std::numeric_limits<std::uint64_t>::max());
	cost[K] = 0;

	std::vector<bool> selected;
	selected.resize(lengths.size(), false);

	while (true) {
		std::uint64_t min_cost  = std::numeric_limits<std::uint64_t>::max();
		std::uint64_t min_index = 0;

		for (std::size_t i = 1; i < cost.size(); ++i) {
			if (!selected[i] && cost[i] < min_cost) {
				min_cost  = cost[i];
				min_index = i;
			}
		}

		if (min_index == 0) break;

		for (auto const & p : lengths[min_index]) {
			if (cost[p.first] >= cost[min_index] + p.second) {
				cost[p.first] = cost[min_index] + p.second;
			}
		}

		selected[min_index] = true;
	}

	return cost;
}

int main() {
	std::uint64_t N, Q, K;
	std::uint64_t a, b, c, x, y;

	std::cin >> N;

	std::vector<umap> lengths;
	lengths.resize(N + 1);

	for (std::uint64_t i = 1; i < N; ++i) {
		std::cin >> a >> b >> c;
		lengths[a][b] = lengths[b][a] = c;
	}

	std::cin >> Q >> K;

	std::vector<uint64_t> cost = dijkstra(K, lengths);
	for (std::uint64_t i = 0; i < Q; ++i) {
		std::cin >> x >> y;
		std::cout << cost[x] + cost[y] << std::endl;
	}

	return 0;
}
