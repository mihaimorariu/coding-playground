#include <array>
#include <iostream>

int main() {
	std::string S;
	std::cin >> S;

	std::array<int, 26> count = {{0}};

	for (auto const c : S) ++count[c - 'a'];

	std::size_t i = 0;
	for (i = 0; i < 26; ++i) {
		if (!count[i]) break;
	}

	if (i == 26) {
		std::cout << "None" << std::endl;
	} else {
		std::cout << static_cast<char>('a' + i) << std::endl;
	}

	return 0;
}
