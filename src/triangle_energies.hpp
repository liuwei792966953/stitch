// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include "energy.hpp"


class TriangleOrthoStrain : public Energy {
    using Matrix3x2 = Eigen::Matrix<double,3,2>;
    using Vector6d  = Eigen::Matrix<double,6,1>;

    public:
    TriangleOrthoStrain(const Eigen::Vector3i& idxs,
            const std::vector<Eigen::Vector3d> &x, double ks);

    int dim() const { return 6; }
    double weight() const { return weight_; }

    void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const;

    void project(Eigen::VectorXd& zi) const;

    Eigen::VectorXd reduce(const Eigen::VectorXd& x) const;

    protected:
    Eigen::Vector3i idxs_;

    double A_;
    double weight_;
    Eigen::Matrix2d rest_;

    Eigen::Matrix<double,3,2> S_;
};

