#include <iostream>

int main() {
	std::string S;
	std::cin >> S;

	for (std::size_t i = 0; i < S.length(); ++i) {
		std::cout << S.substr(0, i + 1) << std::endl;
	}

	return 0;
}
