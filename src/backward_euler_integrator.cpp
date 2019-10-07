
#include "backward_euler_integrator.hpp"

#include "self_collision_energy.hpp"
#include "timer.hpp"



void BackwardEulerIntegrator::initialize(const TriMesh& mesh, double dt) {

    std::cout << "Precomputing" <<std::endl;
    std::vector<Eigen::Triplet<double>> triplets;

    for (int i=0; i<mesh.n_vertices(); i++) {
        for (int j=0; j<3; j++) {
            triplets.push_back(SparseTripletd(3*i+j, 3*i+j, 1.0));
        }
    }

    for (const auto& energy : energies) {
        energy->precompute(mesh);
        energy->getHessianPattern(mesh, triplets);
    }
    std::cout << "Precomputing done" <<std::endl;

    for (int idx=0; idx<mesh.f.rows(); idx++) {
        int idxs[3] = { mesh.f(idx,0), mesh.f(idx,1), mesh.f(idx,2) };
        for (size_t j=0; j<3; ++j)
            for (size_t k=0; k<3; ++k)
                for (size_t l=0; l<3; ++l)
                    for (size_t n=0; n<3; ++n)
                        triplets.push_back(SparseTripletd(3*idxs[j]+l,3*idxs[k]+n, 1.0));
    }

    dFdx_ = SparseMatrixd(mesh.x.size(), mesh.x.size());
    dFdx_.setFromTriplets(triplets.begin(), triplets.end());
    dFdv_ = SparseMatrixd(mesh.x.size(), mesh.x.size());
    dFdv_.setFromTriplets(triplets.begin(), triplets.end());

    F_  = VecXd::Zero(mesh.x.size());
    dv_ = VecXd::Zero(mesh.x.size());
    dx_ = VecXd::Zero(mesh.x.size());
    resid_ = VecXd::Zero(mesh.x.size());

    cg_.resize(mesh.x.rows() / 3);
    std::cout << "initialization done" <<std::endl;
}

bool BackwardEulerIntegrator::is_stitching_phase(TriMesh& mesh, double tol) {
    if (mesh.s.rows() == 0) {
        return true;
    }

    int closed = 0;
    for (int i=0; i<mesh.s.rows(); i++) {
        closed += ((mesh.x.segment<3>(3*mesh.s(i,0)) -
                    mesh.x.segment<3>(3*mesh.s(i,1))).norm() < tol) ? 1 : 0;
    }

    return (double(closed) / double(mesh.s.rows())) < 0.85;
}

void BackwardEulerIntegrator::step(TriMesh& mesh, double dt, int internal_iters, double kd, double mu) {
    //const Eigen::Vector3d gravity(0.0, 0.0, 0.0);
    const Eigen::Vector3d gravity(0.0, -980.0, 0.0);
    const double offset = 0.2;

    static int iteration = 0;


    Timer timer;
    timer.start("Total");

    if (F_.size() != mesh.x.rows()) {
        timer.start("Initialization");
        initialize(mesh, dt);
        timer.stop("Initialization");
    } else {
        if (avatar) {
            timer.start("Avatar update");
            avatar->next_frame(dt);
            timer.stop("Avatar update");
        }
    }

    iteration++;
    std::cout << "Iteration: " << iteration << std::endl;

    SelfCollisionForce self_forces;

    timer.start("Compute forces");
    {
        for (int i=0; i<dFdx_.outerSize(); i++) {
            for (SparseMatrixd::InnerIterator it(dFdx_, i); it; ++it) {
                it.valueRef() = 0.0;
            }
        }
        for (int i=0; i<dFdv_.outerSize(); i++) {
            for (SparseMatrixd::InnerIterator it(dFdv_, i); it; ++it) {
                it.valueRef() = 0.0;
            }
        }

        dx_.setZero();
        dv_.setZero();

        if (!is_stitching_phase(mesh) || iteration > 15 || !avatar) {
            std::cout << "Draping phase" << std::endl;
            for (int i=0; i<mesh.x.size() / 3; i++) {
                F_.segment<3>(3*i) = mesh.m.segment<3>(3*i).array() * gravity.array();
            }
        } else {
            std::cout << "Stitching phase" << std::endl;
            F_.setZero();
            mesh.v.setZero();
        }

        for (const auto& energy : energies) {
            energy->getForceAndHessian(mesh, mesh.x, F_, dFdx_, dFdv_);
        }

        // Damping
        for (int i=0; i<mesh.f.rows(); i++) {
            Eigen::Vector3d n = (mesh.x.segment<3>(3*mesh.f(i,1)) - mesh.x.segment<3>(3*mesh.f(i,0))).cross(mesh.x.segment<3>(3*mesh.f(i,2)) - mesh.x.segment<3>(3*mesh.f(i,0)));
        
            const double kd = -0.01 * 0.01 * 0.5 * n.norm() / 3.0;
            n.normalize();
            for (int j=0; j<3; j++) {
                int idx = mesh.f(i,j);
                F_.segment<3>(3*idx) += n * kd * n.dot(mesh.v.segment<3>(3*idx));
                for (int k=0; k<3; k++) {
                    for (int l=0; l<3; l++) {
                        dFdv_.coeffRef(3*idx+k,3*idx+l) += n[k] * n[l] * kd;
                    }
                }
            }
        }

        if (iteration < 15) {
            const double kd = 40.0 - double(iteration);
            //const double kd = 1.0;
            for (int i=0; i<mesh.e.rows(); i++) {
                int i0 = mesh.e(i, 0);
                int i1 = mesh.e(i, 1);

                Vec3d f = (mesh.v.segment<3>(3*i0) - mesh.v.segment<3>(3*i1)) * kd; 
                F_.segment<3>(3*i0) -= f;
                F_.segment<3>(3*i1) += f;

                for (size_t j=0; j<3; j++)
                {
                    dFdv_.coeffRef(3*i0+j, 3*i0+j) -= kd;
                    dFdv_.coeffRef(3*i1+j, 3*i1+j) -= kd;
                    dFdv_.coeffRef(3*i0+j, 3*i1+j) += kd;
                    dFdv_.coeffRef(3*i1+j, 3*i0+j) += kd;
                }
            }
        }

        timer.start("Self collisions");
        const bool do_layer_intersections = true;
        if (do_layer_intersections) {
            mesh.bvh.refit(mesh.f, mesh.x, 2.0);
            self_forces.getForceAndHessian(mesh, mesh.x, F_, dFdx_, dFdv_);
        }
        timer.stop("Self collisions");

        timer.start("Avatar collisions");
        if (avatar) {
            cg_.reset();

            timer.start("Detection");
            auto cs = Collisions::get_avatar_collisions(*avatar, mesh.x, offset);
            timer.stop("Detection");

            timer.start("Processing");
            avatar_force_.compute(mesh, *avatar, cs, mesh.x, offset,
                    F_, dFdx_, dFdv_, dx_, dv_, resid_,
                    [&](int idx, const Eigen::Matrix3d& S) { cg_.setFilter(idx, S); }, mu, dt);
            timer.stop("Processing");

        }
        timer.stop("Avatar collisions");

        for (const auto& idx : pins_) {
            cg_.setFilter(idx, Eigen::Matrix3d::Zero());
        }
    }
    timer.stop("Compute forces");

    timer.start("Form system and rhs");
    
        rhs_ = (F_ + dFdx_ * (mesh.v * dt + dx_)) * dt;

        // We will re-use the space in dFdx to save expensive allocations
        // Using a reference to make it clear we are now forming the system
        // matrix A
        // A = M + dt * dFdv + dt * dt * dFdx
        SparseMatrixd& A = dFdx_;
        A *= dt;
        A += dFdv_;
        A *= -dt;

        // TODO: Maybe Eigen has some fast way to add a diagonal?
        for (int i=0; i<mesh.m.size(); i++) {
            A.coeffRef(i,i) += mesh.m[i];
        }

    timer.stop("Form system and rhs");

    timer.start("Solve");
    {
        auto multiply = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            VecXd out = A * x;
            self_forces.multiply(x, -dt*dt, -dt, out);
            return out;
        };

        //dv_.setZero(); // TODO: Better initial guess?
        cg_.compute(A);
        cg_.solve(multiply, rhs_, dv_);

        mesh.v.noalias() += dv_;
        mesh.x.noalias() += mesh.v * dt + dx_;

        mesh.update_normals();
        
        resid_ = A * dv_ - rhs_;
    }
    timer.stop("Solve");

    timer.stop("Total");
    timer.summary();
}
