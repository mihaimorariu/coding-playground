#include <iostream>

int main() {
	std::string s;
	std::cin >> s;

	std::size_t start = s.find('A');
	std::size_t end   = s.rfind('Z');

	std::cout << end - start + 1 << std::endl;

	return 0;
}
