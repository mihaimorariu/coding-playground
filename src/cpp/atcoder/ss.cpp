#include <iostream>

bool isStringEven(std::string const & str) {
	if (str.length() % 2 == 1) return false;

	std::size_t length = str.length();
	return str.substr(0, length / 2) == str.substr(length / 2, length / 2);
}

int main() {
	std::string S;
	std::cin >> S;

	for (std::size_t i = S.length() - 1; i >= 1; --i) {
		std::string str = S.substr(0, i);
		if (isStringEven(str)) {
			std::cout << str.length() << std::endl;
			break;
		}
	}

	return 0;
}
