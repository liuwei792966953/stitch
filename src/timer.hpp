#pragma once

#include <chrono>


class Timer
{
public:
	Timer() { reset(); }

	void reset() { start_ = std::chrono::steady_clock::now(); }

	double elapsed() {
	    curr_ = std::chrono::steady_clock::now();
	    std::chrono::duration<double, std::milli> diff = curr_ - start_;
	    return diff.count();
	}

private:
	std::chrono::time_point<std::chrono::steady_clock> start_;
	std::chrono::time_point<std::chrono::steady_clock> curr_;

};


