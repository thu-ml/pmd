#include "utils.h"
#include <cmath>

bool near(float a, float b) {
    return fabs(a - b) < 1e-5;
}

std::uniform_real_distribution<float> u01;
std::mt19937 generator;
