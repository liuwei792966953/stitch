// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include "collisions.hpp"
#include "energy.hpp"



class SelfCollisionEnergy : public DynamicEnergy {
public:
    using Vector9d = Eigen::Matrix<double,9,1>;

    SelfCollisionEnergy(int i0, int i1, int i2, int i3, bool flipped)
        : flipped_(flipped) {
        idxs_[0] = i0;
        idxs_[1] = i1;
        idxs_[2] = i2;
        idxs_[3] = i3;
        weights_ = Vector9d::Constant(100.0);
    }

    SelfCollisionEnergy() = delete;
    SelfCollisionEnergy(const SelfCollisionEnergy&) = default;
    SelfCollisionEnergy(SelfCollisionEnergy&&) = default;
    ~SelfCollisionEnergy() = default;

    int dim() const { return 9; }

    Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
        Vector9d vec;
        vec << x.segment<3>(3*idxs_[1]) - x.segment<3>(3*idxs_[0]),
               x.segment<3>(3*idxs_[2]) - x.segment<3>(3*idxs_[0]),
               x.segment<3>(3*idxs_[3]) - x.segment<3>(3*idxs_[0]);

        return vec;
    }
    
    void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const {
        for (int i=0; i<3; i++) {
            triplets.push_back(Eigen::Triplet<double>(i,   3*idxs_[1],  1.0));
            triplets.push_back(Eigen::Triplet<double>(i,   3*idxs_[0], -1.0));
            triplets.push_back(Eigen::Triplet<double>(3+i, 3*idxs_[2],  1.0));
            triplets.push_back(Eigen::Triplet<double>(3+i, 3*idxs_[0], -1.0));
            triplets.push_back(Eigen::Triplet<double>(6+i, 3*idxs_[3],  1.0));
            triplets.push_back(Eigen::Triplet<double>(6+i, 3*idxs_[0], -1.0));
        }
    }
    
    void project(Eigen::VectorXd &zi) const {
        static const double h = 0.5;

        const Eigen::Vector3d n = (zi.segment<3>(3) - zi.head<3>()).normalized() * (flipped_ ? -1.0 : 1.0);
        const double d = zi.tail<3>().dot(n);

        if (d < h) {
            zi.tail<3>() += n * (h - d);
        }
    }

    void multiply(const Eigen::VectorXd& x, const Eigen::VectorXd& factor, const Eigen::VectorXd& shift,
                    Eigen::VectorXd& out) const {
        Vector9d Dx = reduce(x).array() * factor.array();
        Eigen::Matrix<double, 12, 1> prod = Dt * Dx;
        for (int i=0; i<4; i++) {
            out.segment<3>(3*idxs_[i]).noalias() += prod.segment<3>(3*i) +
                    x.segment<3>(3*idxs_[i]) * shift[idxs_[i]];
        }
    }

    void do_something(const Vector9d& z, double dt, Eigen::VectorXd& b) const {
        Eigen::Matrix<double,12,1> Dt_z = Dt * (z.array() * weights_.array() * weights_.array()).matrix() * dt * dt;
        for (size_t i=0; i<4; i++) {
            b.segment<3>(3*idxs_[i]).noalias() += Dt_z.segment<3>(3*i);
        }
    }

    const Eigen::Vector4i indices() const { return idxs_; }

protected:
    Eigen::Vector4i idxs_;

    bool flipped_;
    
private:
    static Eigen::Matrix<double, 12, 9> Dt;
};

class SelfCollisionForce : public BaseEnergy
{
public:
    struct CollisionData
    {
        Eigen::Matrix3d nnT;
        Eigen::Vector4i idxs;
        Eigen::Vector4d w;
    };
    
    void precompute(const TriMesh& mesh) { }

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const {
	collisions_.clear();

        const double h = 0.5;

        std::mutex collision_mutex;

        mesh.bvh.self_intersect([&](int v, int f) {
                if (mesh.f(f,0) == v ||
                    mesh.f(f,1) == v ||
                    mesh.f(f,2) == v) {
                    return;
                }

                if (mesh.vl[v] == mesh.fl[f]) {
                    return;
                }

                Eigen::Vector3d w;
                if (Collisions::get_barycentric_weights(mesh.x.segment<3>(3*v),
                                                        mesh.x.segment<3>(3*mesh.f(f,0)),
                                                        mesh.x.segment<3>(3*mesh.f(f,1)),
                                                        mesh.x.segment<3>(3*mesh.f(f,2)), w)) {

                    Vec3d n = mesh.fn.segment<3>(3*f);

                    bool flipped = mesh.fl[f] > mesh.vl[v];
                    if (mesh.fl[f] > mesh.vl[v]) {
                        n *= -1.0;
                    }

                    double d = (mesh.x.segment<3>(3*v) -
                                mesh.x.segment<3>(3*mesh.f(f,0))).dot(n);
                    if (d < h && d > -2.5 && n.dot(mesh.vn.segment<3>(3*v)) > 0.0) {
                        CollisionData c;
                        c.idxs << v, mesh.f(f,0), mesh.f(f,1), mesh.f(f,2);
                        c.w << 1.0, -w;
                        
                        c.nnT = n * n.transpose();
                        
                        const Vec3d rel_vel = mesh.v.segment<3>(3*c.idxs[0]) -
                                                mesh.v.segment<3>(3*c.idxs[1]) * w[0] -
                                                mesh.v.segment<3>(3*c.idxs[2]) * w[1] -
                                                mesh.v.segment<3>(3*c.idxs[3]) * w[2];

                        const double vn = n.dot(rel_vel);
                        const Vec3d vt = rel_vel - n * vn;

                        Mat3d ortho = Mat3d::Identity() - c.nnT;

                        #pragma omp critical
                        {
                        collisions_.emplace_back(c);

                        for (int i=0; i<4; i++) {
                            F.segment<3>(3*c.idxs[i]) += (n * ((h - d) * ks_ - vn * kd_) - vt * mu_) * c.w[i];

                            for (int j=0; j<3; j++) {
                                for (int k=0; k<3; k++) {
                                    dFdx.coeffRef(3*c.idxs[i]+j, 3*c.idxs[i]+k) += c.nnT(j,k) * ks_ * c.w[i] * c.w[i];
                                    dFdv.coeffRef(3*c.idxs[i]+j, 3*c.idxs[i]+k) -= (c.nnT(j,k) * kd_ + ortho(j,k) * mu_) * c.w[i] * c.w[i];
                                }
                            }
                        }
                        }
                    }
                }
        });

        std::cout << "Found " << collisions_.size() << " collisions." << std::endl;
    }

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd>& triplets) const { }

    void multiply(const VecXd& in, const double dFdx_factor, const double dFdv_factor, VecXd& out) const {
        for (const CollisionData& c : collisions_) {
            Mat3d ortho = Mat3d::Identity() - c.nnT;

            // Each collision contributes 16 3x3 matrix blocks
            // The diagonals are already accounted for, so we
            // implicity construct the remaining 12
            for (int i=0; i<4; i++) {
                for (int j=0; j<4; j++) {
                    if (i == j) { continue; } // Skip diagonals

                    // The i-j dFdx matrix looks like nnT * ks_ * w[i] * w[j]
                    // dFdv looks like nnT * kd_ * w[i] * w[j]
                    out.segment<3>(3*c.idxs[i]).noalias() += c.nnT * in.segment<3>(3*c.idxs[j]) * ks_ * c.w[i] * c.w[j] * dFdx_factor;
                    out.segment<3>(3*c.idxs[i]).noalias() -= (c.nnT * kd_ + ortho * mu_) * in.segment<3>(3*c.idxs[j]) * c.w[i] * c.w[j] * dFdv_factor;
                }
            }
        }
    }

protected:
    const double ks_ = 500.0;
    const double kd_ =   25.0;

    // TODO: Should be passed in
    const double mu_ = 0.1;

    mutable std::vector<CollisionData> collisions_;
};
