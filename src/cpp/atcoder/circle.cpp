#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
	std::uint16_t W, H, x, y, r;
	std::cin >> W >> H >> x >> y >> r;

	for (std::uint16_t i = 0;  i < H; ++i) {
		for (std::uint16_t j = 0; j < W; ++j) {
			std::uint16_t length = std::round(std::sqrt(std::pow(i - y, 2) + std::pow(j - x, 2)));
			std::cout << std::setw(2) << (length == r ? "#" : ".");
		}
		std::cout << std::endl;
	}

	return 0;
}
