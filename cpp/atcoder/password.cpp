#include <iostream>

int main() {
	std::string O, E;
	std::cin >> O >> E;

	for (std::size_t i = 0; i < E.size(); ++i) {
		if (i == O.size()) {
			std::cout << E[i];
		} else {
			std::cout << O[i] << E[i];
		}
	}

	std::cout << std::endl;

	return 0;
}
