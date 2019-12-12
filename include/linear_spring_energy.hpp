// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include "energy.hpp"



class LinearSpringEnergy : public BaseEnergy
{
public:
    LinearSpringEnergy(double ksx, double) : ks_(ksx) { }

    void precompute(const TriMesh& mesh) { lambda_ = VecXd(mesh.e.rows()); }

    virtual void getForceAndHessian(const TriMesh& mesh,
                                    const Eigen::VectorXd& x,
				    Eigen::VectorXd& F,
				    SparseMatrixd& dFdx,
				    SparseMatrixd& dFdv) const {
	assert(0);
    }

    virtual void getHessianPattern(const TriMesh& mesh,
                                   std::vector<SparseTripletd> &triplets) const {

        for (int i=0; i<mesh.e.rows(); i++) {
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    triplets.push_back({ 3*mesh.e(i,0)+j, 3*mesh.e(i,1)+k, 1.0 });
                }
            }
        }
    }


    // XPBD
    size_t nbrEnergies(const TriMesh& mesh) const { return mesh.e.rows(); }

    void perVertexCount(const TriMesh& mesh, std::vector<int>& counts) const {
        for (int idx=0; idx<mesh.e.rows(); idx++) {
            for (int j=0; j<2; j++) {
                counts[mesh.e(idx,j)]++;
            }
        }
    }

    void update(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) {
        for (int i=0; i<mesh.e.rows(); i++) {
            const int i0 = mesh.e(i, 0);
            const int i1 = mesh.e(i, 1);

            const double L = (mesh.u.segment<2>(2*i0) - mesh.u.segment<2>(2*i1)).norm();

            Vec3d n = x.segment<3>(3*i0) - x.segment<3>(3*i1);
            const double l = n.norm();

            if (l > L) {
                const double a = (1.0 / ks_) / (dt * dt);
                const double C = l - L;

                const double dl = (-C - lambda_[i] * a) /
                    ((1.0 / mesh.m[3*i0]) + (1.0 / mesh.m[3*i1]) + a);

                lambda_[i] += dl;
                n /= l;

                dx.segment<3>(3*i0) += n * dl / mesh.m[3*i0];
                dx.segment<3>(3*i1) -= n * dl / mesh.m[3*i1];
            }
        }
    }
    
    void reset() { lambda_.setZero(); }

protected:
    VecXd lambda_; // For XPBD

    const double ks_ = 1.0e6;
};


template <typename MeshT>
struct LinearSpring
{
    using ElementT = typename MeshT::Edge;

    static constexpr int dim = 1;

    static int n_constraints() { return 1; }

    static typename MeshT::Real ks(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.stretch_ks(it);
    }
    
    static typename MeshT::Real kd(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.stretch_kd(it);
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
        
        const auto L = (u1 - u0).norm(); // Rest length
        const auto l = n.norm();         // Current length

        dC0 = -n;
        dC1 =  n;

        C[0] = l - L;

        if (l > 1.0e-8) {
            dC0 /= l;
            dC1 /= l;
        }
    }
};
