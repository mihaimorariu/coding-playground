#include <cmath>
#include <iostream>
#include <tuple>

template<typename... T>
struct arity {
	int constexpr static value = sizeof...(T);
};

template<typename ... T>
void foo(T ... args) {
}

int main() {
	auto my_tuple = std::make_tuple(51, "Frans Nielsen", 1.2f);

	std::cout << std::get<0>(my_tuple) << std::endl;
	std::cout << std::get<1>(my_tuple) << std::endl;
	std::cout << std::get<2>(my_tuple) << std::endl;
	std::cout << "----------" << std::endl;

	std::string name;
	int age;
	std::tie(std::ignore, age, name) = std::make_tuple(0, 29, "John James");

	std::cout << name << std::endl;
	std::cout << age << std::endl;
	std::cout << "----------" << std::endl;

	return 0;
}
