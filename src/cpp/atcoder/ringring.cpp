#include <iostream>

int main() {
	std::uint16_t a, b, c;
	std::cin >> a >> b >> c;

	if (a + b < a + c) {
		if (a + b < b + c) {
			std::cout << a + b << std::endl;
		} else {
			std::cout << b + c << std::endl;
		}
	} else if (a + c < b + c) {
		std::cout << a + c << std::endl;
	} else {
		std::cout << b + c << std::endl;
	}

	return 0;
}
