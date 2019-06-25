// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include "admm_integrator.hpp"

#include "timer.hpp"


void ADMM_Integrator::initialize(const TriMesh& mesh, double dt) {
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> weights;
    for (const auto& energy : energies) {
        std::vector<Eigen::Triplet<double>> ts;
        energy->get_reduction(ts);

        for (const auto& t : ts) {
            triplets.push_back(Eigen::Triplet<double>(t.row() + weights.size(), t.col(), t.value()));
        }

        energy->set_index(weights.size());
        std::copy_n(energy->weights().data(), energy->dim(), std::back_inserter(weights));
    }

    u_ = Eigen::VectorXd::Zero(weights.size());
    z_ = Eigen::VectorXd::Zero(weights.size());

    // Create the Selector+Reduction matrix
    SparseMatrixd D(weights.size(), mesh.x.rows());
    D.setFromTriplets(triplets.begin(), triplets.end());

    // Set global matrices
    SparseMatrixd W2dt2(weights.size(), weights.size());
    W2dt2.reserve(weights.size());
    for (size_t i=0; i<weights.size(); ++i) {
        W2dt2.coeffRef(i,i) = weights[i] * weights[i] * dt * dt;
    }

    DtWtW_ = D.transpose() * W2dt2;
    A_ = SparseMatrixd(DtWtW_ * D);
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


void ADMM_Integrator::step(TriMesh& mesh, double dt, int internal_iters, double kd, double mu) {
        const Eigen::Vector3d gravity(0.0, -980.0, 0.0);

        Timer timer;
        timer.start("Total");

        if (DtWtW_.rows() != mesh.x.rows()) {
            std::cout << "Initializing..." << std::endl;
            timer.start("Initialization");
            initialize(mesh, dt);
            timer.stop("Initialization");
            std::cout << "Done." << std::endl;
        } else {
            if (avatar) {
                timer.start("Avatar update");
                avatar->next_frame(dt);
                timer.stop("Avatar update");
            }
        }

        // Temporary hack to figure out how long to wait until turning gravity on
        static int iteration = 0;
        
        #pragma omp parallel for
        for (size_t i=0; i<energies.size(); i++) {
            energies[i]->update(iteration);
        }

        // TODO: Have a "stitching" phase defined if they are mostly closed or not
        // TODO: Other external forces
        if (iteration > 10) {
            for (int i=0; i<mesh.v.size() / 3; i++) {
                mesh.v.segment<3>(3*i) += cg_.S(i) * gravity * dt;
            }
        } else {
            mesh.v.setZero();
        }
        iteration++;

        timer.start("Explicit step");
       
        // Explicit step forward by dt
        Eigen::VectorXd x_curr = mesh.x + mesh.v * dt;
        
        timer.stop("Explicit step");

        timer.start("Z-update");

        #pragma omp parallel for num_threads(4)
        for (size_t i=0; i<energies.size(); i++) {
            z_.segment(energies[i]->index(), energies[i]->dim()) = energies[i]->reduce(mesh.x);
        }
        
        timer.stop("Z-update");

        // Element-wise multiplication (m.asDiagonal() * x_curr would also work)
        const Eigen::VectorXd Mx = mesh.m.array() * x_curr.array();

        dx_ = Eigen::VectorXd::Zero(mesh.x.rows());

        timer.start("Internal iterations");

        for (int iter=0; iter<internal_iters; iter++) {
            timer.start("Local energy update");
            #pragma omp parallel for num_threads(4)
            for (size_t i=0; i<energies.size(); i++) {
                const auto& energy = energies[i];

                Eigen::VectorXd Dix = energy->reduce(x_curr);
                Eigen::VectorXd zi = Dix + u_.segment(energy->index(), energy->dim());
                energy->project(zi);

                u_.segment(energy->index(), energy->dim()).noalias() += Dix - zi;
                z_.segment(energy->index(), energy->dim()).noalias() = zi;
	    }
            timer.stop("Local energy update");

            timer.start("Solve / global update");
	    const Eigen::VectorXd b = Mx + DtWtW_ * (z_ - u_);

	    cg_.solve(A_, b, x_curr);
	    //x_curr = ldlt_.solve(Mx + DtWtW_ * (z_ - u_));

            dx_.noalias() += A_ * x_curr - b;
            timer.stop("Solve / global update");
        }

        timer.stop("Internal iterations");

        mesh.v = (x_curr - mesh.x) * (1.0 - kd) / dt;
        mesh.x = x_curr;

        timer.start("Avatar collisions");
 
        if (avatar) {
            cg_.reset();

            timer.start("Detection");

            const double offset = 0.2;
            auto cs = Collisions::get_avatar_collisions(*avatar, mesh.x, offset);

            timer.stop("Detection");

            timer.start("Processing");
            
            #pragma omp parallel for
            for (int i=0; i<mesh.x.size() / 3; i++) {
                if (cs[i].tri_idx != -1) {
                    Eigen::Vector3d avatar_vel = Eigen::Vector3d::Zero();
                    for (int j=0; j<3; j++) {
                        avatar_vel.noalias() += avatar->v.segment<3>(3*avatar->f(cs[i].tri_idx,j)) * cs[i].w[j];
                    }

                    cs[i].rel_vel = mesh.v.segment<3>(3*i) - avatar_vel;

                    bool colliding = true;
                    if (collisions_[i].tri_idx != -1) {
                        if (cs[i].dx > 0.8 * offset &&
                                dx_.segment<3>(3*i).dot(collisions_[i].n) < 0.0) {
                            colliding = false;
                        }
                    } else if (cs[i].dx > 0.0 &&
                            (offset - cs[i].dx) < cs[i].rel_vel.dot(collisions_[i].n) * dt) {
                        colliding = false;
                    }

                    if (colliding) {
                        collisions_[i] = cs[i];
                    } else {
                        collisions_[i].tri_idx = -1;
                    }
                } else {
                    collisions_[i].tri_idx = -1;
                }

                if (collisions_[i].tri_idx != -1) {
                    double dx = collisions_[i].dx < 0.0 ? 0.05 * offset - collisions_[i].dx
                                                        : 0.5 * (offset * 0.8 - collisions_[i].dx);

                    const Eigen::Vector3d& n = collisions_[i].n;

                    if (collisions_[i].dx < 0.8 * offset) {
                        mesh.x.segment<3>(3*i) += n * dx;
                    }

                    cg_.setFilter(i, Eigen::Matrix3d::Identity() - n * n.transpose());

                    const double vn = collisions_[i].rel_vel.dot(n);
                    if (vn < 0.0) {
                        mesh.v.segment<3>(3*i) -= n * vn;
                    }

                    const double mag = dx_.segment<3>(3*i).dot(n);
                    //const double mag = dx + dx_.segment<3>(3*i).dot(n);
                    //const double mag = -vn;
                    if (mag > 0.0) {
                        Eigen::Vector3d vt = collisions_[i].rel_vel - n * vn;
                        const double vt_mag = vt.norm();

                        const double dv_max = mu * mag;// / dt;
                        
                        const Eigen::Vector3d t = vt / vt_mag;

                        if (vt_mag < dv_max) {
                            // Static friction
                            mesh.v.segment<3>(3*i) -= vt;
                            //cg_.setFilter(i, Eigen::Matrix3d::Zero());
                    cg_.setFilter(i, Eigen::Matrix3d::Identity() - n * n.transpose() - t * t.transpose());
                        } else {
                            // Dynamic friction
                            // Subtract out as much velocity as allowed by
                            // Coulomb's law
                            mesh.v.segment<3>(3*i) -= t * dv_max;
                        }
                    }
                }
            }

            timer.stop("Processing");
        }

        timer.stop("Avatar collisions");

        timer.summary();
        //const double ms = timer.elapsed();
        //std::cout << (1000.0 / ms) << " fps (" << ms << " ms)" << std::endl;
    }

