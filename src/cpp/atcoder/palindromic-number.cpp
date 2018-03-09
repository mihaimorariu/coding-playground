#include <iostream>

int main() {
	std::uint16_t N;
	std::cin >> N;

	std::cout << (N / 100 == N % 10 ? "Yes" : "No") << std::endl;

	return 0;
}
