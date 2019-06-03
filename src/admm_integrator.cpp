#include "admm_integrator.hpp"

#include "timer.hpp"


void ADMM_Integrator::initialize(const SimMesh& mesh, double dt) {
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> weights;
    for (const auto& energy : energies) {
        energy->get_reduction(triplets, weights);
    }

    u_ = Eigen::VectorXd::Zero(weights.size());

    // Create the Selector+Reduction matrix
    D_.resize(weights.size(), mesh.x.rows());
    D_.setZero();
    D_.setFromTriplets(triplets.begin(), triplets.end());

    // Set global matrices
    SparseMatrixd W2dt2(weights.size(), weights.size());
    W2dt2.reserve(weights.size());
    for (size_t i=0; i<weights.size(); ++i) {
        W2dt2.coeffRef(i,i) = weights[i] * weights[i] * dt * dt;
    }

    DtWtW_ = D_.transpose() * W2dt2;
    A_ = SparseMatrixd(DtWtW_ * D_);
    for (int i=0; i<mesh.x.rows(); i++) {
        A_.coeffRef(i,i) += mesh.m[i];
    }

    ldlt_.compute(A_);
    cg_.compute(A_);

    collisions_.resize(mesh.x.rows() / 3);

    for (const auto& idx : pins_) {
        cg_.setFilter(idx, Eigen::Matrix3d::Zero());
    }

    dx_ = Eigen::VectorXd::Zero(mesh.x.rows());
}


void ADMM_Integrator::step(SimMesh& mesh, double dt, int internal_iters, double kd, double mu) {
        const Eigen::Vector3d gravity(0.0, -98.0, 0.0);

        if (DtWtW_.rows() != mesh.x.rows()) {
            std::cout << "Initializing..." << std::endl;
            initialize(mesh, dt);
            std::cout << "Done." << std::endl;
        }

        Timer timer;

        // TODO: Other external forces
        for (int i=0; i<mesh.v.size() / 3; i++) {
            mesh.v.segment<3>(3*i) += gravity * dt;
        }

        if (avatar) {
            cg_.reset();

            #pragma omp parallel for
            for (int i=0; i<mesh.x.size() / 3; i++) {
                Collision c;
                c.dx = 0.05;

                avatar->bvh.visit(mesh.x.segment<3>(3*i),
                        [&](int idx) {
                            Eigen::RowVector3d w;
                            igl::barycentric_coordinates(mesh.x.segment<3>(3*i).transpose(),
                                            avatar->x.segment<3>(3*avatar->f(idx,0)).transpose(),
                                            avatar->x.segment<3>(3*avatar->f(idx,1)).transpose(),
                                            avatar->x.segment<3>(3*avatar->f(idx,2)).transpose(), w);

                            if (std::all_of(w.data(), w.data()+3, [](double v) { return v > -0.001 && v < 1.001; })) {

                            Eigen::Vector3d n = (avatar->x.segment<3>(3*avatar->f(idx,1)) - avatar->x.segment<3>(3*avatar->f(idx,0))).cross(avatar->x.segment<3>(3*avatar->f(idx,2)) - avatar->x.segment<3>(3*avatar->f(idx,0))).normalized();

                            double dist = (mesh.x.segment<3>(3*i) - avatar->x.segment<3>(3*avatar->f(idx,0))).dot(n);
                            if (dist < c.dx) {
                                c.dx = dist;
                                c.tri_idx = idx;
                                c.n = n;
                            }
                            }
                        });

                //if (collisions_[i].tri_idx != -1 && dx_.segment<3>(3*i).dot(collisions_[i].n)*dt > mesh.v.segment<3>(3*i).dot(collisions_[i].n)) {
                if (collisions_[i].tri_idx != -1 && mesh.v.segment<3>(3*i).dot(collisions_[i].n) > 0.0) {
                    //std::cout << "Releasing constraint: " << i << std::endl;
                    collisions_[i].tri_idx = -1;
                }

                if (c.tri_idx != -1) {
                    if (c.dx < 0.0) { c.dx = 0.01 - c.dx; }
                    else { c.dx = 0.8 * (0.05 - c.dx); }

                    collisions_[i] = c;
                }

                if (collisions_[i].tri_idx != -1) {
                    const Eigen::Vector3d& n = collisions_[i].n;
                    mesh.x.segment<3>(3*i) += n * collisions_[i].dx;
                    cg_.setFilter(i, Eigen::Matrix3d::Identity() - n * n.transpose());
                    collisions_[i].dx = 0.0;

                    const double mag = dx_.segment<3>(3*i).dot(n);
                    if (mag < 0.0) {
                        Eigen::Vector3d vn = n * mag / dt;
                        const double dv_max = std::fabs(mag);

                        Eigen::Vector3d vt = mesh.v.segment<3>(3*i) - n * mesh.v.segment<3>(3*i).dot(n);

                        const double vt_mag = vt.norm();

                        if (vt_mag < mu * dv_max) {
                            mesh.v.segment<3>(3*i) -= vt;
                            cg_.setFilter(i, Eigen::Matrix3d::Zero());
                        } else {
                            const Eigen::Vector3d t = vt / vt_mag;

                            mesh.v.segment<3>(3*i) -= t * (vt_mag - mu * dv_max);
                        }
                    }
                }
            }
        }

        // Explicit step forward by dt
        Eigen::VectorXd x_curr = mesh.x + cg_.filter(mesh.v * dt);

        // Replace this with a per-element reduction
        z_ = D_ * mesh.x;

        // Element-wise multiplication (m.asDiagonal() * x_curr would also work)
        const Eigen::VectorXd Mx = mesh.m.array() * x_curr.array();

        //dx_ = Eigen::VectorXd::Zero(mesh.x.rows());

        const int nbr_internal_iters = 10;
        for (int iter=0; iter<nbr_internal_iters; iter++) {
            #pragma omp parallel for
            for (size_t i=0; i<energies.size(); i++) {
	        energies[i]->update(D_, x_curr, z_, u_);
	    }

	    const Eigen::VectorXd b = Mx + DtWtW_ * (z_ - u_);

	    cg_.solve(A_, b, x_curr);
	    //x_curr = ldlt_.solve(Mx + DtWtW_ * (z_ - u_));
            //std::cout << "\t" << x_curr.head<3>().transpose() << std::endl;

            //dx_.noalias() += A_ * x_curr - b;
        }

        mesh.v = (x_curr - mesh.x) * (1.0 - kd) / dt;
        mesh.x = x_curr;

        std::cout << "Timer: " << timer.elapsed() << std::endl;
    }

