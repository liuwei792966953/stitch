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


class StitchSpringEnergy : public BaseEnergy
{
public:
    StitchSpringEnergy(double ks, double kd) : ks_(ks), kd_(kd) {}

    void precompute(const TriMesh& mesh) { }

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const {

	const double ks = ks_ * std::min(10.0, std::pow(1.1, iteration++));

        for (int i=0; i<mesh.s.rows(); i++) {
            const int i0 = mesh.s(i,0);
            const int i1 = mesh.s(i,1);

            // Zero-length spring. Doesn't get any easier...
            Vec3d n = x.segment<3>(3*i1) - x.segment<3>(3*i0);
            const double l = n.norm();
            if (l > 1.0e-8) {
                n /= l;
            }
            Vec3d v = mesh.v.segment<3>(3*i1) - mesh.v.segment<3>(3*i0);

            F.segment<3>(3*i0).noalias() += n * l * ks + v * kd_;
            F.segment<3>(3*i1).noalias() -= n * l * ks + v * kd_;

            for (size_t l=0; l<3; l++) {
                dFdx.coeffRef(3*i0+l, 3*i0+l) -= ks;
                dFdx.coeffRef(3*i1+l, 3*i1+l) -= ks;
                dFdx.coeffRef(3*i0+l, 3*i1+l) += ks;
                dFdx.coeffRef(3*i1+l, 3*i0+l) += ks;

                dFdv.coeffRef(3*i0+l, 3*i0+l) -= kd_;
                dFdv.coeffRef(3*i1+l, 3*i1+l) -= kd_;
                dFdv.coeffRef(3*i0+l, 3*i1+l) += kd_;
                dFdv.coeffRef(3*i1+l, 3*i0+l) += kd_;
            }
        }
    }

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd>& triplets) const {
        for (int i=0; i<mesh.s.rows(); i++) {
            const int i0 = mesh.s(i,0);
            const int i1 = mesh.s(i,1);

            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
		    triplets.push_back(SparseTripletd(3*i0+j,3*i0+k, 1.0));
		    triplets.push_back(SparseTripletd(3*i1+j,3*i1+k, 1.0));
		    triplets.push_back(SparseTripletd(3*i0+j,3*i1+k, 1.0));
		    triplets.push_back(SparseTripletd(3*i1+j,3*i0+k, 1.0));
                }
            }
        }
    }
   
protected:
    const Real ks_;
    const Real kd_;

    mutable int iteration = 0;
};

