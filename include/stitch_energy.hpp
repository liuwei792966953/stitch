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

    void precompute(const TriMesh& mesh) override {
        lambda_ = VecXd::Zero(mesh.s.rows());
    }

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const override {

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

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd>& triplets) const override {
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

    void perVertexCount(const TriMesh& mesh,
                        std::vector<int>& counts) const override {
        for (int i=0; i<mesh.s.rows(); i++) {
            for (int j=0; j<2; j++) {
                counts[mesh.s(i,j)]++;
            }
        }
    }
      
protected:
    const double ks_;
    const double kd_;

    mutable int iteration = 0;
};


template <typename MeshT>
struct StitchSpring
{
    using ElementT = typename MeshT::Stitch;

    static constexpr int dim = 1;

    static int n_constraints() { return 1; }

    static typename MeshT::Real ks(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.stitch_ks(it);
    }
    
    static typename MeshT::Real kd(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.stitch_kd(it);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    static void project(const Eigen::MatrixBase<DerivedA>& u0, 
                 const Eigen::MatrixBase<DerivedA>& u1,
                 const Eigen::MatrixBase<DerivedB>& x0,
                 const Eigen::MatrixBase<DerivedB>& x1,
                 Eigen::MatrixBase<DerivedC>& C,
                 Eigen::MatrixBase<DerivedD>& dC0,
                 Eigen::MatrixBase<DerivedD>& dC1)
    {
        auto n = x1 - x0;
        const auto l = n.norm();

        dC1 =  n;
        dC0 = -n;
        if (l > 1.0e-8) {
            dC1 /= l;
            dC0 /= l;
        }
        
        C[0] = l;
    }

    /*
    void project(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) {
	//const double ks = ks_ * std::min(10.0, std::pow(1.1, iteration++));

	double a = 1.0 / ks_;
	a /= dt * dt;

	const double b = dt * dt * kd_;
	const double c = a * b / dt;

        for (int i=0; i<mesh.s.rows(); i++) {
            const int i0 = mesh.s(i,0);
            const int i1 = mesh.s(i,1);

            Vec3d n = mesh.x.segment<3>(3*i1) - mesh.x.segment<3>(3*i0);// +
                          //dx.segment<3>(3*i1) - dx.segment<3>(3*i0); // Jacobi
            const double l = n.norm();
            if (l > 1.0e-8) {
                n /= l;
            }

            double dl = (-l - lambda_[i] * a - c * (n.dot(x.segment<3>(3*i1) - mesh.x.segment<3>(3*i1)) - n.dot(x.segment<3>(3*i0) - mesh.x.segment<3>(3*i0)))) / ((1.0 + c) * (1.0 / mesh.m[3*i0] + 1.0 / mesh.m[3*i1]) + a);
	    lambda_[i] += dl;

            dx.segment<3>(3*i0) -= n * dl / mesh.m[3*i0];
            dx.segment<3>(3*i1) += n * dl / mesh.m[3*i1];
        }
    }
    */

};
