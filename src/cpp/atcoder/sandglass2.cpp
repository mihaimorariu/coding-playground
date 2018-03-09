#include <iostream>

int main() {
	int X, t;

	std::cin >> X >> t;
	std::cout << std::max(X - t, 0) << std::endl;

	return 0;
}
