#include <iostream>
#include <limits>

int main() {
	std::int16_t A, B, C, D;

	std::cin >> A >> B >> C >> D;

	std::int16_t start = std::max(A, C);
	std::int16_t end   = std::min(B, D);
	std::cout << (end - start < 0 ? 0 : end - start) << std::endl;

	return 0;
}
