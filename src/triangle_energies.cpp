// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include "triangle_energies.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>


TriangleOrthoStrain::TriangleOrthoStrain(const Eigen::Vector3i& idxs,
                                         const std::vector<Eigen::Vector3d>& x,
                                         double ks)
    : idxs_(idxs)  {

    Matrix3x2 Ds;
    Ds.col(0) = x[1] - x[0];
    Ds.col(1) = x[2] - x[0];

    // Take n1 as just the first edge, n2 as the orthogonal vector closest to second edge
    Eigen::Vector3d n1 = Ds.col(0).normalized();
    Eigen::Vector3d n2 = (Ds.col(1) - Ds.col(1).dot(n1) * n1).normalized();

    Matrix3x2 Dm;
    Dm.col(0) = n1;
    Dm.col(1) = n2;

    Eigen::Matrix2d F = Dm.transpose() * Ds;
    rest_ = F.inverse();

    A_ = F.determinant() * 0.5;

    weight_ = std::sqrt(ks * A_);

    S_.setZero();
    S_(0,0) = -1; S_(0,1) = -1;
    S_(1,0) =  1; S_(2,1) =  1;
}


void TriangleOrthoStrain::get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const {
    const int cols[3] = { 3*idxs_[0], 3*idxs_[1], 3*idxs_[2] };

    Matrix3x2 D = S_ * rest_;
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            triplets.emplace_back(i, cols[j]+i, D(j,0));
            triplets.emplace_back(3+i, cols[j]+i, D(j,1));
        }
    }
}

Eigen::VectorXd TriangleOrthoStrain::reduce(const Eigen::VectorXd& x) const {
    const int cols[3] = { 3*idxs_[0], 3*idxs_[1], 3*idxs_[2] };
    
    Vector6d z = Vector6d::Zero();

    Matrix3x2 D = S_ * rest_;
    for (int i=0; i<3; ++i ) {
        for (int j=0; j<3; j++) {
            z[i]   += D(j,0) * x[cols[j]+i];
            z[3+i] += D(j,1) * x[cols[j]+i];
        }
    }

    return z;
}

void TriangleOrthoStrain::project(Eigen::VectorXd& zi) const {
    Eigen::JacobiSVD<Matrix3x2>
        svd(Eigen::Map<Matrix3x2>(zi.data()), Eigen::ComputeThinU | Eigen::ComputeThinV);

    Matrix3x2 P = svd.matrixU().leftCols(2) * svd.matrixV().transpose();
    zi = 0.5 * (Eigen::Map<Vector6d>(P.data()) + zi);
}


