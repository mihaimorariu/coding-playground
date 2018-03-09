#include <iostream>

int main() {
	std::uint16_t a, b, c;
	std::cin >> a >> b >> c;

	if (a + c == b || a + b == c || b + c == a) {
		std::cout << "Yes" << std::endl;
	} else {
		std::cout << "No" << std::endl;
	}

	return 0;
}
