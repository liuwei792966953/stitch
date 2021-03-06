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
    cg_.resize(mesh.x.size() / 3);

    collisions_.resize(mesh.x.rows() / 3);

    for (const auto& idx : pins_) {
        cg_.setFilter(idx, Eigen::Matrix3d::Zero());
    }

    dx_ = Eigen::VectorXd::Zero(mesh.x.rows());
}


void ADMM_Integrator::step(TriMesh& mesh, double dt, int internal_iters, double kd, double mu) {
        //const Eigen::Vector3d gravity(0.0, 0.0, 0.0);
        const Eigen::Vector3d gravity(0.0, -980.0, 0.0);
        const double offset = 0.2;

        auto multiply = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd vec = A_ * x;
            //for (const auto& e : collision_energies) {
            //    e.multiply(x, e.weights() * dt * dt, mesh.m, vec);
            //}
            return vec;
        };

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

        dx_.setZero();

        // TODO: Have a "stitching" phase defined if they are mostly closed or not
        // TODO: Other external forces
        if (iteration > 10) {
            for (int i=0; i<mesh.v.size() / 3; i++) {
                mesh.v.segment<3>(3*i) += cg_.S(i) * gravity * dt;
            }
        } else {
            mesh.v.setZero();
        }

        const bool do_layer_intersections = false;

        if (do_layer_intersections) {
        const double ks = 1.0;
        const double h = 0.1;

        collision_energies.clear();

        timer.start("Mesh intersect");
        mesh.bvh.refit(mesh.f, mesh.x, 2.0);
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
                    Eigen::Vector3d n = 
                            (mesh.x.segment<3>(3*mesh.f(f,1)) - mesh.x.segment<3>(3*mesh.f(f,0))).cross(mesh.x.segment<3>(3*mesh.f(f,2)) - mesh.x.segment<3>(3*mesh.f(f,0))).normalized();

                    bool flipped = mesh.fl[f] > mesh.vl[v];
                    if (mesh.fl[f] > mesh.vl[v]) {
                        n *= -1.0;
                    }

                    double d = (mesh.x.segment<3>(3*v) - mesh.x.segment<3>(3*mesh.f(f,0))).dot(n);
                    if (d < h) {
                        collision_energies.emplace_back(v, mesh.f(f,0), mesh.f(f,1), mesh.f(f,2), flipped);
                        //std::cout << "\tCollision: " << v << " <-> " << f << "; " << d << "; " << mesh.fl[f] << "; " << mesh.vl[v] << std::endl;
                        //mesh.v.segment<3>(3*v) -= dt * ks * (h - d) * n / mesh.m[3*v];
                        //for (size_t i=0; i<3; i++) {
                        //    mesh.v.segment<3>(3*mesh.f(f,i)) += dt * ks * w[i] * (h - d) * n / mesh.m[3*mesh.f(f,i)];
                        //}
                    }
                }
        });
        timer.stop("Mesh intersect");

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

        Eigen::VectorXd u_collision = Eigen::VectorXd::Zero(9*collision_energies.size());

        Eigen::VectorXd z_collision(9*collision_energies.size());
        #pragma omp parallel for num_threads(4)
        for (size_t i=0; i<collision_energies.size(); i++) {
            z_collision.segment<9>(9*i) = collision_energies[i].reduce(mesh.x);
        }
        
        timer.stop("Z-update");

        // Element-wise multiplication (m.asDiagonal() * x_curr would also work)
        const Eigen::VectorXd Mx = mesh.m.array() * x_curr.array();

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
	    
	    Eigen::VectorXd b = Mx + DtWtW_ * (z_ - u_);

            timer.stop("Local energy update");

            timer.start("Collision energy update");
            #pragma omp parallel for num_threads(4)
            for (size_t i=0; i<collision_energies.size(); i++) {
                const auto& energy = collision_energies[i];

                Eigen::VectorXd Dix = energy.reduce(x_curr);
                Eigen::VectorXd zi = Dix + u_collision.segment<9>(9*i);
                energy.project(zi);

                u_collision.segment<9>(9*i).noalias() += Dix - zi;
                z_collision.segment<9>(9*i).noalias() = zi;
            }
            for (size_t i=0; i<collision_energies.size(); i++) {
                collision_energies[i].do_something(z_collision.segment<9>(9*i) -
                                                 u_collision.segment<9>(9*i), dt, b);
            }
            timer.stop("Collision energy update");

            timer.start("Solve / global update");

	    //cg_.solve(multiply, b, x_curr);
	    x_curr = ldlt_.solve(b);

            dx_.noalias() += A_ * x_curr - b;
            timer.stop("Solve / global update");
        }

        timer.stop("Internal iterations");

        timer.start("Avatar collisions");

        if (avatar) {
            cg_.reset();

            timer.start("Detection");

            auto cs = Collisions::get_avatar_collisions(*avatar, x_curr, offset);

            timer.stop("Detection");

            timer.start("Processing");
            
            #pragma omp parallel for
            for (int i=0; i<mesh.x.size() / 3; i++) {
                if (cs[i].tri_idx != -1) {
                    Eigen::Vector3d avatar_vel = Eigen::Vector3d::Zero();
                    for (int j=0; j<3; j++) {
                        avatar_vel.noalias() += avatar->v.segment<3>(3*avatar->f(cs[i].tri_idx,j)) * cs[i].w[j];
                    }

                    cs[i].av_dx = avatar_vel * dt;

                    bool colliding = true;
                    if (collisions_[i].tri_idx != -1) {
                        if (cs[i].dx > 0.8 * offset &&
                                dx_.segment<3>(3*i).dot(collisions_[i].n) < 0.0) {
                            colliding = false;
                        }
                    }
                    else if (cs[i].dx > 0.0 &&
                            (offset - cs[i].dx) <
                                //(mesh.v.segment<3>(3*i) * dt -
                                 ((x_curr.segment<3>(3*i) - mesh.x.segment<3>(3*i)) -
                                    cs[i].av_dx).dot(cs[i].n)) {
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
                        x_curr.segment<3>(3*i) += n * dx;
                    }
                
                    cg_.setFilter(i, Eigen::Matrix3d::Identity() - n * n.transpose());

                    dx_.segment<3>(3*i) = n * dx;
                } else {
                    dx_.segment<3>(3*i).setZero();
                }
            }

            timer.stop("Processing");
        }

        timer.stop("Avatar collisions");

        timer.start("Avatar friction");
 
        if (avatar) {
            #pragma omp parallel for
            for (int i=0; i<mesh.x.size() / 3; i++) {
                const Eigen::Vector3d dx_i = x_curr.segment<3>(3*i) - mesh.x.segment<3>(3*i);

                if (collisions_[i].tri_idx != -1) {
                    const Eigen::Vector3d& n = collisions_[i].n;

                    const double mag = dx_.segment<3>(3*i).dot(n);
                    if (mag > 0.0) {
                        Eigen::Vector3d rel_dx = dx_i - collisions_[i].av_dx;

                        Eigen::Vector3d dx_t = rel_dx - n * n.dot(rel_dx);
                        const double dx_mag = dx_t.norm();

                        const double dx_max = mu * mag;

                        if (dx_mag < dx_max) {
                            // Static friction
                            //x_curr.segment<3>(3*i) = mesh.x.segment<3>(3*i);
                            x_curr.segment<3>(3*i) -= dx_t;
                            cg_.setFilter(i, Eigen::Matrix3d::Zero());
                        } else {
                            // Dynamic friction
                            // Subtract out as much velocity as allowed by
                            // Coulomb's law
                            x_curr.segment<3>(3*i) -= dx_t * dx_max / dx_mag;
                        }
                    }
                }
            }
        }

        timer.stop("Avatar friction");

        mesh.v = cg_.filter(x_curr - mesh.x) * (1.0 - kd) / dt;
        mesh.x = x_curr;

        timer.summary();
    }

