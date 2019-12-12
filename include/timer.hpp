// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <chrono>
#include <string>
#include <vector>


class Stopwatch
{
public:
    Stopwatch() : diff_(0) { start(); }

    void start() {
        start_ = std::chrono::steady_clock::now();
        running_ = true;
    }

    void stop() {
        diff_ += std::chrono::steady_clock::now() - start_;
        running_ = false;
    }

    double elapsed() {
        if (running_) {
            diff_ = std::chrono::steady_clock::now() - start_;
        }
        return diff_.count();
    }

    bool running() const {
        return running_;
    }

private:
    bool running_ = false;

    std::chrono::time_point<std::chrono::steady_clock> start_;
    std::chrono::duration<double, std::milli> diff_;
};


class Timer
{
public:
    bool start(const std::string& name) {
        auto it = std::find(names_.begin(), names_.end(), name);
        if (it == names_.end()) {
            int nbr_running = std::count_if(clocks_.begin(), clocks_.end(),
                    [&](const auto& sw) { return sw.running(); });

            names_.push_back(name);
            clocks_.push_back(Stopwatch());
            depths_.push_back(nbr_running);

            return false;
        }
        
        // Resume already existing stopwatch
        clocks_[std::distance(names_.begin(), it)].start();

        return true;
    }

    bool stop(const std::string& name) {
        auto it = std::find(names_.begin(), names_.end(), name);
        if (it == names_.end()) {
            // No such clock
            return false;
        }

        clocks_[std::distance(names_.begin(), it)].stop();
;
        return true;
    }

    void summary() {
        std::cout << "----------------" << std::endl;
        for (size_t i=0; i<names_.size(); i++) {
            const std::string fill(2*depths_[i], '-');
            std::cout << fill << " " << names_[i] << " " << clocks_[i].elapsed() << std::endl;
        }
        //std::cout << "----------------" << std::endl;
    }

    void clear() {
        names_.clear();
        clocks_.clear();
        depths_.clear();
    }

private:
    std::vector<std::string> names_;
    std::vector<Stopwatch>  clocks_;
    std::vector<int>        depths_;
};
