#include <iostream>

int main() {
	std::uint16_t A, B, C;
	std::cin >> A >> B >> C;

	bool haiku = (A == 5 && B == 5 && C == 7) ||
	             (A == 5 && B == 7 && C == 5) ||
	             (A == 7 && B == 5 && C == 5);

	std::cout << (haiku ? "YES" : "NO") << std::endl;

	return 0;
}
