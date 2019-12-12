// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <iostream>
#include <Eigen/StdVector>

#include "energy.hpp"

class ConstrainedJacobi {
public:
    bool compute(const SparseMatrixd& A) {
        if (A.rows() != A.cols()) {
            std::cerr << "Non-square matrix A!" << std::endl;
            return false;
        }

        S_.resize(A.rows() / 3, Eigen::Matrix3d::Identity());

        return true;
    }

    void reset() {
        for (Eigen::Matrix3d& M : S_) {
            M.setIdentity();
        }
    }

    void setFilter(int idx, const Eigen::Matrix3d& C) {
        S_[idx] = C;
    }

    void filterInPlace(Eigen::VectorXd& v) {
        #pragma omp parallel for
        for (size_t i=0; i<S_.size(); i++) {
            v.segment<3>(3*i) = S_[i] * v.segment<3>(3*i);
        }
    }

    Eigen::VectorXd filter(const Eigen::VectorXd& v) {
        Eigen::VectorXd out(v.size());
        #pragma omp parallel for
        for (size_t i=0; i<S_.size(); i++) {
            out.segment<3>(3*i) = S_[i] * v.segment<3>(3*i);
        }
        return out;
    }

    void solve(const SparseMatrixd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {

        Eigen::VectorXd x_tmp(A.rows());

        const int max_iters = 200;
        const double max_error = 1.e0-5;
        double error = 1.0;
        int iter = 0;
        while (iter < max_iters && error > max_error) {
            error = 0.0;
            #pragma omp parallel for
            for (int i=0; i<A.rows()/3; i++) {
                for (int j=0; j<3; j++) {
                    int idx = 3 * i + j;

                    double omega = 0.0;
                    double a_ii;

                    for (SparseMatrixd::InnerIterator rit(A, idx) ; rit; ++rit) {
                        if (rit.col() == idx) {
                            a_ii = rit.value();
                        } else {
                            omega += rit.value() * x[rit.col()];
                        }
                    }

                    x_tmp[idx] = (b[idx] - omega) / a_ii;
                    error += (x_tmp[idx] - x[idx]) * (x_tmp[idx] - x[idx]);
                }

                x.segment<3>(3*i) += S_[i] * (x_tmp.segment<3>(3*i) - x.segment<3>(3*i));
            }

            //error = (A*x - b).norm();
            iter++;
        }

        //std::cout << "Finished in " << iter << " with error " << error << std::endl;

    }

    const Eigen::Matrix3d& S(int idx) {
        return S_[idx];
    }

protected:
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> S_;
};


