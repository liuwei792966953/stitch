// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <iostream>
#include <Eigen/StdVector>

#include "energy.hpp"


class ModifiedConjugateGradient {
public:
    bool compute(const SparseMatrixd& A) {
        if (A.rows() != A.cols()) {
            std::cerr << "Non-square matrix A!" << std::endl;
            return false;
        }

        if (P_.size() != A.rows() / 3) {
            P_.resize(A.rows() / 3);
        }

        for (int i=0; i<A.rows(); i+=3) {
            Mat3d Ai = A.block(i,i,3,3);
            P_[i/3] = Ai.inverse();
        }
        
        return true;
    }

    void resize(int nbr) {
        S_.resize(nbr, Eigen::Matrix3d::Identity());
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

    template <typename F, typename Derived>
    void solve(F multiply, const Eigen::MatrixBase<Derived>& b, Eigen::VectorXd& x) {
        static double eps = 2.5e-3;
        static const size_t maxNbrItrs = 1000;

        VecXd r = b - multiply(x); filterInPlace(r);
        VecXd c(b.size());

        for (int i=0; i<b.size()/3; i++) {
            c.segment<3>(3*i) = S_[i] * P_[i] * r.segment<3>(3*i);
        }

        double dNew = r.dot(c);
        double d0 = dNew;

        double tol = eps * eps * d0;

        size_t nbrItrs = 0;
        size_t minNbr = 10;
        while ((dNew > tol || nbrItrs < minNbr) && nbrItrs < maxNbrItrs)
        {
            VecXd q = multiply(c); filterInPlace(q);
            double a = dNew / c.dot(q);

            x.noalias() += a * c;
            r.noalias() -= a * q;
            
            VecXd s(r.size());
            for (int i=0; i<b.size()/3; i++) {
                s.segment<3>(3*i) = P_[i] * r.segment<3>(3*i);
            }

            double dOld = dNew;
            dNew = r.dot(s);

            c *= dNew / dOld;
            c.noalias() += s;
	    filterInPlace(c);

            nbrItrs++;
        }

        std::cout << "CG finished in " << nbrItrs
                  << " iterations with error " << dNew << " (" << tol << ")" << std::endl;
    }

    const Eigen::Matrix3d& S(int idx) {
        return S_[idx];
    }

protected:
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> P_;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> S_;
};


