#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <tuple>

using namespace std::chrono_literals;

long long operator "" _celsius(unsigned long long celsius) {
	return std::llround(celsius * 1.8 + 32);
}

long long operator ""_fahrenheit(unsigned long long fahrenheit) {
	return std::llround((fahrenheit - 32) / 1.8);
}

auto f1(int i) { return i; };

template<typename T>
auto & f2(T & t) { return t; };

int main() {
	std::cout << 25_celsius << std::endl;
	std::cout << 75_fahrenheit << std::endl;
	std::cout << "----------" << std::endl;

	auto day = 24h;
	std::cout << day.count() << std::endl;
	std::cout << "----------" << std::endl;

	std::cout << 0b1111'1111 << std::endl;
	std::cout << "----------" << std::endl;

	auto identity = [](auto x) { return x; };
	std::cout << identity(3) << std::endl;
	std::cout << identity("foo") << std::endl;
	std::cout << "----------" << std::endl;

	auto generator = [x = 0]() mutable { return ++x; };
	std::cout << generator() << std::endl;
	std::cout << generator() << std::endl;
	std::cout << generator() << std::endl;
	std::cout << "----------" << std::endl;

	auto p = std::make_unique<int>(1);
	//auto task1 = [=] { *p = 5; };
	auto task2 = [p = std::move(p)] { *p = 5; };
	std::cout << "----------" << std::endl;

	auto x = 1;
	auto f = [&r = x, x = x * 10] { ++r; return r + x; };
	std::cout << f() << std::endl;
	std::cout << "----------" << std::endl;

	auto g = [](auto & x) -> auto & { return f2(x); };
	int y = 123;
	int & z = g(y);
	std::cout << z << std::endl;
	std::cout << "----------" << std::endl;

	return 0;
}
