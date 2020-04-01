#include <iostream>
#include <variant>

int main() {
	std::variant<std::int8_t> v{10};
	//std::cout << std::get<int>(v) << std::endl;
	//std::cout << std::get<0>(v) << std::endl;
	//v = 12.5;

	//std::cout << std::get<double>(v) << std::endl;
	//std::cout << std::get<1>(v) << std::endl;
	//v = 11.f;

	//std::cout << std::get<float>(v) << std::endl;
	//std::cout << std::get<2>(v) << std::endl;
	//std::cout << sizeof << std::endl;

	std::cout << sizeof(v) << std::endl;

	return 0;
}
