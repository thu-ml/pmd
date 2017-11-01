#include <iostream>
#include <vector>
#include <random>

bool near(float a, float b);

template<class T>
std::ostream& operator << (std::ostream &out, const std::vector<T> &a) {
    for (auto &c: a)
        out << c << ' ';
    return out;
}

extern std::uniform_real_distribution<float> u01;
extern std::mt19937 generator;

