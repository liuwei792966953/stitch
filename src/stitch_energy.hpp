// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include "energy.hpp"



class StitchEnergy : public Energy {
public:
    StitchEnergy(int idx1, int idx2)
        : idx1_(idx1), idx2_(idx2) {
        weights_ = Eigen::VectorXd::Constant(3, 10000.0);
    }

    int dim() const { return 3; }

    Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
        return x.segment<3>(3*idx2_) - x.segment<3>(3*idx1_);
    }

    void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const {
        for (size_t i=0; i<3; i++) {
            triplets.emplace_back(i, 3*idx2_+i,  1.0);
            triplets.emplace_back(i, 3*idx1_+i, -1.0);
        }
    }

    void project(Eigen::VectorXd &zi) const {
        //zi = Eigen::Vector3d::Zero();
        zi *= kd_;
    }
    
    virtual void update(int iter) {
        if (iter > 25) { kd_ = 0.5; }
        else if (iter > 15) { kd_ = 0.75; }
    }

protected:
    int idx1_;
    int idx2_;

    double kd_ = 0.95;

};
