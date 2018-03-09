#include <iostream>
#include <map>

int main() {
	std::uint32_t N;
	std::uint64_t area = 0, A;

	std::map<std::uint64_t, std::uint64_t> occurences;
	std::cin >> N;

	for (std::uint32_t i = 0; i < N; ++i) {
		std::cin >> A;

		if (!occurences.count(A)) {
			occurences.insert({A, 1});
		} else {
			++occurences[A];
		}
	}

	for (auto it = occurences.rbegin(); it != occurences.rend(); ++it) {
		if (it->second >= 2) {
			std::uint64_t L = static_cast<std::uint64_t>(it->first);

			if (it->second >= 4) {
				area = area > 0 ? area * L : L * L;
				break;
			} else {
				if (area) {
					area *= L;
					break;
				} else {
					area = L;
				}
			}

		}
	}

	std::cout << area << std::endl;

	return 0;
}
