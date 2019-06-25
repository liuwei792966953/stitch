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

        //P_ = Eigen::VectorXd(A.rows());

        //for (int i=0; i<A.rows(); i++) {
        //    if (A.coeff(i,i) < 1.0e-12) {
        //        std::cerr << "Negative entry on matrix diagonal!" << std::endl;
        //        return false;
        //    }
        //    P_[i] = 1.0 / A.coeff(i,i);
     
        //}

        P_.resize(A.rows() / 3);
        for (int i=0; i<A.rows(); i+=3) {
            Eigen::Matrix3d Ai = A.block(i,i,3,3);
            P_[i/3] = Ai.inverse();
        }
        
        //P_ = SparseMatrixd(A.rows(), A.rows());

	//std::vector<Eigen::Triplet<double>> triplets;
	//for (int i=0; i<A.rows(); i+=3) {
        //    Eigen::Matrix3d Ai = A.block(i,i,3,3);
        //    Eigen::Matrix3d A_inv = Ai.inverse();
        //    for (int j=0; j<3; j++)
        //        for (int k=0; k<3; k++)
        //            triplets.push_back(Eigen::Triplet<double>(i+j,i+k,Ai(j,k)));
        //}

        //P_.setFromTriplets(triplets.begin(), triplets.end());

	//Eigen::VectorXi nnz = Eigen::VectorXi::Ones(A.rows());
	//P_.reserve(nnz);
	//for (int i=0; i<A.rows(); ++i) { P_.coeffRef(i,i) = 1.0 / A.coeff(i,i); }

        S_.resize(A.rows() / 3, Eigen::Matrix3d::Identity());
	//Eigen::VectorXi nnz = Eigen::VectorXi::Ones(A.rows());
	//S_.reserve(nnz);
	//for (int i=0; i<A.rows(); ++i) { S_.coeffRef(i,i) = 1.0; }

        return true;
    }

    void reset() {
        for (Eigen::Matrix3d& M : S_) {
            M.setIdentity();
        }

        //for (int i=0; i<S_.rows() / 3; i++)
        //    S_.block<3,3>(3*i,3*i) = Eigen::Matrix3d::Identity();
    }

    void setFilter(int idx, const Eigen::Matrix3d& C) {
        S_[idx] = C;
        //S_.block<3,3>(3*idx,3*idx) = C;
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

        Eigen::VectorXd r = b - A * x;
        filterInPlace(r);
        //Eigen::VectorXd d = P_.array() * r.array();
        //Eigen::VectorXd d = P_ * r;
        Eigen::VectorXd d(A.rows());
        for (int i=0; i<A.rows()/3; i++) {
            d.segment<3>(3*i) = S_[i] * P_[i] * r.segment<3>(3*i);
        }
        //filterInPlace(d);

        Eigen::VectorXd s(A.rows());

        double d_new = r.dot(d);
        const double d0 = d_new;
        const double eps2 = 1.0e-4;
        const double err = std::max(eps2 * d0, 1.0e-8);

        const int i_max = 1000;
        for (int i=0; i<i_max; i++) {
            Eigen::VectorXd q = filter(A * d);
            double alpha = d_new / d.dot(q);

            x.noalias() += d * alpha;
            r.noalias() -= q * alpha;

            //Eigen::VectorXd s = P_.array() * r.array();
            //Eigen::VectorXd s = P_ * r;
            for (int i=0; i<A.rows()/3; i++) {
                s.segment<3>(3*i) = P_[i] * r.segment<3>(3*i);
            }

            double d_old = d_new;
            d_new = r.dot(s);
            if (d_new < err) {
                //std::cout << "CG: " << i << " / " << i_max << "; " << d_new << " < " << (eps2*d0) << "; " << err << std::endl;
                break;
            }

            d *= d_new / d_old;
            d.noalias() += s;
            filterInPlace(d);
        }
    }

    const Eigen::Matrix3d& S(int idx) {
        return S_[idx];
    }

protected:
    //Eigen::VectorXd P_;
    //SparseMatrixd P_;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> P_;

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> S_;
    //SparseMatrixd S_;
};


