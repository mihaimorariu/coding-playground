#include <iostream>

int main() {
	std::uint16_t A, B;
	std::cin >> A >> B;

	if (A % 3 == 0 || B % 3 == 0 || (A + B) % 3 == 0) {
		std::cout << "Possible" << std::endl;
	} else {
		std::cout << "Impossible" << std::endl;
	}

	return 0;
}
