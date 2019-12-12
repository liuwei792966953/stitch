
#pragma once


using Real = double;

constexpr Real operator"" _r(float v)  { return Real{v}; }
constexpr Real operator"" _r(double v) { return Real{v}; }
