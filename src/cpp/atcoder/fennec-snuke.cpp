#include <iostream>
#include <vector>

using uint32vec = std::vector<uint32_t>;

std::uint32_t getNotSelectedNodes(std::vector<uint32vec> const & adjacents,
                                  std::vector<char>      const & selected,
                                  std::uint32_t          const node)
{
	std::uint32_t count = 0;
	for (auto const & a : adjacents[node]) count += !selected[a];
	return count;
}

std::uint32_t performMove(std::vector<uint32vec>     const & adjacents,
                          std::vector<char>          const & selected,
                          std::vector<std::uint32_t> const & colored)
{
	std::uint32_t next_index = 0;
	std::uint32_t max_adjacents_nodes = 0;

	for (auto const & b : colored) {
		for (auto const & a : adjacents[b]) {
			if (!selected[a]) {
				std::uint32_t count = getNotSelectedNodes(adjacents, selected, a);
				if (count > max_adjacents_nodes) {
					max_adjacents_nodes = count;
					next_index          = a;
				}
			}
		}
	}

	return next_index;
}

int main() {
	std::uint32_t N;
	std::cin >> N;

	std::vector<uint32vec> adjacents(N + 1);
	std::vector<char> selected(N + 1, 0);

	uint32vec blacks;
	uint32vec whites;

	std::uint32_t a, b;
	for (std::uint32_t i = 0; i < N - 1; ++i) {
		std::cin >> a >> b;
		adjacents[a].push_back(b);
		adjacents[b].push_back(a);
	}

	selected[1] = 1;
	selected[N] = 2;

	blacks.push_back(1);
	whites.push_back(N);

	std::string winner = "";
	std::uint32_t next_index = 0;

	while (true) {
		next_index = performMove(adjacents, selected, blacks);

		if (next_index == 0) {
			winner = "Snuke";
			break;
		} else {
			selected[next_index] = 1;
			blacks.push_back(next_index);
		}

		next_index = performMove(adjacents, selected, whites);

		if (next_index == 0) {
			winner = "Fennec";
			break;
		} else {
			selected[next_index] = 2;
			whites.push_back(next_index);
		}
	}

	std::cout << winner << std::endl;

	return 0;
}
