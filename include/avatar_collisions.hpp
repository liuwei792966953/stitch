
#pragma once

#include "collisions.hpp"
#include "energy.hpp"
#include <Eigen/StdVector>



class AvatarCollisionsEnergy
{
public:
    template <typename FilterFunc>
    void compute(const TriMesh& mesh,
                 const AnimatedMesh& avatar,
                 const std::vector<Collision>& cs,
                 const Eigen::VectorXd& x, double offset,
                 Eigen::VectorXd& F,
                 SparseMatrixd& dFdx,
                 SparseMatrixd& dFdv,
                 Eigen::VectorXd& dx,
                 Eigen::VectorXd& dv,
                 Eigen::VectorXd& resid,
                 FilterFunc set_filter,
                 double mu, double dt) {

        const int n_verts = x.size() / 3;

        if (was_previously_colliding_.size() != n_verts) {
            was_previously_colliding_.resize(n_verts, false);
        }

        if (spring_anchors_.size() != n_verts) {
            spring_anchors_.resize(n_verts);
        }

        #pragma omp parallel for
        for (int i=0; i<n_verts; i++) {
            if (cs[i].tri_idx != -1 && cs[i].dx < offset) {
                const Eigen::Vector3d& n = cs[i].n;

                Eigen::Vector3d avatar_vel = Eigen::Vector3d::Zero();
                for (int j=0; j<3; j++) {
                    avatar_vel.noalias() += avatar.v.segment<3>(3*avatar.f(cs[i].tri_idx,j)) * cs[i].w[j];
                }

                bool colliding = true;

                /*
                // If it was previously colliding, but is now outside
                // the offset "band" _and_ the previous step was holding it
                // back, then let this one go
                if (was_previously_colliding_[i]) {
                    if (cs[i].dx > 0.8 * offset &&
                            resid.segment<3>(3*i).dot(n) > 0.0) {
                        colliding = false;
                    }
                }
                // Outside the avatar and the difference will be made
                // up by the relative velocity
                else if (cs[i].dx > 0.0 &&
                        (offset - cs[i].dx) <
                        (mesh.v.segment<3>(3*i) * dt -
                         avatar_vel).dot(n)) {
                    colliding = false;
                }
                */

                if (colliding) {
                    F.segment<3>(3*i) += cs[i].n * ((offset - cs[i].dx) * ks_ - mesh.v.segment<3>(3*i).dot(n) * kd_);
                    for (int j=0; j<3; j++) {
                        for (int k=0; k<3; k++) {
                            dFdx.coeffRef(3*i+j, 3*i+k) -= n[j] * n[k] * ks_;
                            dFdv.coeffRef(3*i+j, 3*i+k) -= n[j] * n[k] * kd_;
                        }
                    }

                    /*
                    double d = cs[i].dx < 0.0 ? 0.05 * offset - cs[i].dx
                        : 0.5 * (offset * 0.8 - cs[i].dx);

                    set_filter(i, Eigen::Matrix3d::Identity() - n * n.transpose());

                    dx.segment<3>(3*i) = n * d;
                    if (mesh.v.segment<3>(3*i).dot(n) < avatar_vel.dot(n)) {
                        dv.segment<3>(3*i) = n.dot(avatar_vel - mesh.v.segment<3>(3*i)) * n;
                    }
                    */

                    if (mu > 1.0e-8)
                    {
                        //const double r_i = resid.segment<3>(3*i).dot(n);
                        const double r_i = (offset - cs[i].dx) * ks_ - mesh.v.segment<3>(3*i).dot(n) * kd_;

                        Vec3d vt = (mesh.v.segment<3>(3*i) - cs[i].n * cs[i].n.dot(mesh.v.segment<3>(3*i)));
                        const double vt_l = vt.norm();
                        if (vt_l > mu * r_i) {
                            vt.normalize();
                        F.segment<3>(3*i) -= mu * r_i * vt;
                        for (int j=0; j<3; j++) {
                            for (int k=0; k<3; k++) {
                                dFdv.coeffRef(3*i+j,3*i+k) -= vt[j] * vt[k] * mu * r_i;
                            }
                        }
                        }

                        /*
                        if (!was_previously_colliding_[i]) {
                            spring_anchors_[i] = x.segment<3>(3*i) + cs[i].n * (offset - cs[i].dx);
                        } else if (r_i > 0.0) {
                            Vec3d spring_vec = mesh.x.segment<3>(3*i) - spring_anchors_[i];

                            // Find the tangential portion of the spring
                            Vec3d spring_tang = spring_vec - n * n.dot(spring_vec);

                            // mu is the friction spring stiffness, so this is the force
                            //Vec3d f_fric = mu * spring_tang;

                            // If the the sliding force exceeds the normal force, then this
                            // is the dynamic friction case, and we need to update the anchor
                            //const double Ft = f_fric.norm();

                            Vec3d nt = spring_tang.normalized();

                            //Vec3d f_fric = nt * mu * mesh.v.segment<3>(3*i).dot(nt);

                            //if (Ft > Fn) {
                            //if (std::fabs(mesh.v.segment<3>(3*i).dot(nt)) > mu * std::fabs(r_i)) {
                            if (std::fabs(mesh.v.segment<3>(3*i).dot(nt)) > mu * std::fabs(r_i)) {
                                Vec3d f_fric = mu * std::fabs(r_i) * nt;
                                F.segment<3>(3*i) -= f_fric;
                                for (int j=0; j<3; j++) {
                                    for (int k=0; k<3; k++) {
                                        dFdv.coeffRef(3*i+j,3*i+k) -= nt[j] * nt[k] * mu;
                                    }
                                }

                                // The dx needed to make Fn == Ft
                                spring_anchors_[i] += nt * ((std::fabs(mesh.v.segment<3>(3*i).dot(nt)) - std::fabs(r_i)) / mu);
                            } else {
                                // Static friction. Just freeze the vertex
                                set_filter(i, Eigen::Matrix3d::Zero());
                            }
                        }
                        */

                        /*
                        if (!was_previously_colliding_[i]) {
                            // If relative velocity is low, pin it
                            Eigen::Vector3d rel_vel = mesh.v.segment<3>(3*i) - avatar_vel;
                            double vn = n.dot(rel_vel);

                            Eigen::Vector3d vt = rel_vel - n * vn;
                            double Fn = n.dot(F.segment<3>(3*i)) * dt / mesh.m[i];

                            if (vt.norm() < mu * std::fabs(Fn)) {
                                set_filter(i, Eigen::Matrix3d::Zero());
                            }
                        } else {
                            // If tangential force exceeds some fraction of normal force
                            // Allow to slide
                            double Ft = (F.segment<3>(3*i) - n * F.segment<3>(3*i).dot(n)).norm();

                            if (Ft < mu * std::fabs(r_i)) {
                                set_filter(i, Eigen::Matrix3d::Zero());
                            } else {
                                // For "high" sliding velocity, dissipative tangential force
                                // proportional to normal force
                                Eigen::Vector3d rel_vel = mesh.v.segment<3>(3*i) - avatar_vel;
                                double vn = n.dot(rel_vel);
                                Eigen::Vector3d vt = rel_vel - n * vn;

                                if (vt.norm() > mu * std::fabs(r_i)) {
                                    F.segment<3>(3*i) -= vt.normalized() * std::fabs(r_i) * mu;
                                }
                            }
                        }
                        */

                    /*
                        const double r_i = resid.segment<3>(3*i).dot(n);

                        if (r_i < 0.0 && was_previously_colliding_[i]) {
                            Eigen::Vector3d rel_vel = mesh.v.segment<3>(3*i) - avatar_vel;
                            double vn = n.dot(rel_vel);
                            Eigen::Vector3d vt = rel_vel - n * vn;
                            const double vt_mag = vt.norm();

                            if (vt_mag > 3.0 * mu * std::fabs(r_i))
                                continue;

                            const double dv_max = mu * std::fabs(r_i);
                            if (vt_mag > dv_max)
                                continue;

                            //Vec3d Ft = F_.segment<3>(3*i) - n * n.dot(F_.segment<3>(3*i));
                            //if (Ft.norm() > mu * std::fabs(Fn) * mesh.m[i] / dt)
                            //    continue;

                            set_filter(i, Eigen::Matrix3d::Zero());
                            dv.segment<3>(3*i) = -rel_vel;
                        }
                        */
                    }
                    
                    was_previously_colliding_[i] = true;

                } else {
                    was_previously_colliding_[i] = false;
                }

            } else {
                was_previously_colliding_[i] = false;
            }
        }
    }

    size_t nbrEnergies(const TriMesh& mesh) const { return mesh.x.size() / 3; }

    void update(const TriMesh& mesh,
                const VecXd& x, 
                 const AnimatedMesh& avatar,
                 const std::vector<Collision>& cs,
                 double offset, double dt,
                 VecXd& dx) {

        const int n_verts = mesh.x.size() / 3;

        if (lambda_.size() != n_verts) {
            lambda_ = VecXd::Zero(n_verts);
        }

        const double a = (1.0 / ks_) / (dt * dt);

	const double b = dt * dt * kd_;
	const double c = a * b / dt;

        #pragma omp parallel for
        for (int i=0; i<n_verts; i++) {
            if (cs[i].tri_idx != -1 && cs[i].dx < offset) {
                const Eigen::Vector3d& n = cs[i].n;

                const double C = (x.segment<3>(3*i) - avatar.x.segment<3>(3*avatar.f(cs[i].tri_idx,0))).dot(n) - offset;

                double dl = (-C - a * lambda_[i] - c * n.dot(x.segment<3>(3*i) - mesh.x.segment<3>(3*i))) / (1.0 / mesh.m[3*i] + a);
                dx.segment<3>(3*i) += n * dl / mesh.m[3*i];

                lambda_[i] += dl;
            }
        }
    }

    void reset() { lambda_.setZero(); }

    void friction(const TriMesh& mesh,
                  const std::vector<Collision>& cs,
                  double mu,
                  VecXd& x) const {
        const int n_verts = mesh.x.size() / 3;

        // Follow Coulomb's Law: dx_t <= mu * dx_n,
        #pragma omp parallel for
        for (int i=0; i<n_verts; i++) {
            if (cs[i].tri_idx != -1) {
                const Vec3d dx = x.segment<3>(3*i) - mesh.x.segment<3>(3*i);
                
                const double dx_n = dx.dot(cs[i].n);
                Vec3d dx_ortho = dx - cs[i].n * dx.dot(cs[i].n);

                const double dx_t = std::min(dx_ortho.norm(), mu * std::fabs(dx_n));

                x.segment<3>(3*i) -= dx_ortho.normalized() * dx_t;
            }
        }
    }

protected:
    const double ks_ = 1.0e6;
    const double kd_ = 10.0;

    std::vector<bool> was_previously_colliding_;
    
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> spring_anchors_;

    VecXd lambda_;
};
