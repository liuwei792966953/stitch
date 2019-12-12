// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include "avatar_collisions.hpp"
#include "collisions.hpp"
#include "energy.hpp"
#include <typeinfo>
#include <typeindex>
#include <unordered_map>


struct XPBD_Integrator
{
    template <typename MeshT>
    static bool is_stitching_phase(const MeshT& mesh, double tol=0.5) {
        const int n_stitches = mesh.template n_elements<typename MeshT::Stitch>();
        if (n_stitches == 0) {
            return true;
        }

        int closed = 0;
        for (auto it=mesh.template begin<typename MeshT::Stitch>();
                 it!=mesh.template end<typename MeshT::Stitch>(); ++it) {
            const auto vs = mesh.stitch_vertices(it);
            closed += ((mesh.x(vs[1]) - mesh.x(vs[0])).norm() < tol) ? 1 : 0;
        }

        return (double(closed) / double(n_stitches)) < 0.85;
    }


    template <typename MeshT>
    static void init_lambdas(MeshT& mesh, std::unordered_map<std::type_index, typename MeshT::VecXr>& lambdas) { }

    template <typename MeshT, typename EnergyT, typename... Energies>
    static void init_lambdas(MeshT& mesh, std::unordered_map<std::type_index, typename MeshT::VecXr>& lambdas) {
        lambdas.emplace(std::type_index(typeid(EnergyT)), MeshT::VecXr::Zero(mesh.template n_elements<typename EnergyT::ElementT>() * EnergyT::n_constraints()));

        init_lambdas<MeshT, Energies...>(mesh, lambdas);
    }
    
    template <typename MeshT, typename EnergyT, int N>
    struct project_n {
        static void _(MeshT& mesh, const typename MeshT::VecXr& x,
                const std::array<typename MeshT::Vertex::iterator, N>& vs,
                Eigen::Matrix<typename MeshT::Real, EnergyT::dim, 1>& C,
                std::array<Eigen::Matrix<typename MeshT::Real, 3, EnergyT::dim>, N>& dC)
                { assert(false && "Unsupported N!"); }
    };

    template <typename MeshT, typename EnergyT>
    struct project_n<MeshT, EnergyT, 1> {
        static void _(MeshT& mesh, const typename MeshT::VecXr& x,
                const std::array<typename MeshT::Vertex::iterator, 1>& vs,
                Eigen::Matrix<typename MeshT::Real, EnergyT::dim, 1>& C,
                std::array<Eigen::Matrix<typename MeshT::Real, 3, EnergyT::dim>, 1>& dC)
        {
            EnergyT::project(mesh.u(vs[0]), x.template segment<3>(3*mesh.vidx(vs[0])), C, dC[0]);
        }
    };

    template <typename MeshT, typename EnergyT>
    struct project_n<MeshT, EnergyT, 2> {
        static void _(MeshT& mesh, const typename MeshT::VecXr& x,
                const std::array<typename MeshT::Vertex::iterator, 2>& vs,
                Eigen::Matrix<typename MeshT::Real, EnergyT::dim, 1>& C,
                std::array<Eigen::Matrix<typename MeshT::Real, 3, EnergyT::dim>, 2>& dC)
        {
            EnergyT::project(mesh.u(vs[0]),
                             mesh.u(vs[1]),
                             x.template segment<3>(3*mesh.vidx(vs[0])),
                             x.template segment<3>(3*mesh.vidx(vs[1])),
                             C, dC[0], dC[1]);
        }
    };

    template <typename MeshT, typename EnergyT>
    struct project_n<MeshT, EnergyT, 3> {
        static void _(MeshT& mesh, const typename MeshT::VecXr& x,
                const std::array<typename MeshT::Vertex::iterator, 3>& vs,
                Eigen::Matrix<typename MeshT::Real, EnergyT::dim, 1>& C,
                std::array<Eigen::Matrix<typename MeshT::Real, 3, EnergyT::dim>, 3>& dC)
        {
            EnergyT::project(mesh.u(vs[0]),
                             mesh.u(vs[1]),
                             mesh.u(vs[2]),
                             x.template segment<3>(3*mesh.vidx(vs[0])),
                             x.template segment<3>(3*mesh.vidx(vs[1])),
                             x.template segment<3>(3*mesh.vidx(vs[2])),
                             C, dC[0], dC[1], dC[2]);
        }
    };

    template <typename MeshT, typename EnergyT>
    struct project_n<MeshT, EnergyT, 4> {
        static void _(MeshT& mesh, const typename MeshT::VecXr& x,
                const std::array<typename MeshT::Vertex::iterator, 4>& vs,
                Eigen::Matrix<typename MeshT::Real, EnergyT::dim, 1>& C,
                std::array<Eigen::Matrix<typename MeshT::Real, 3, EnergyT::dim>, 4>& dC)
        {
            EnergyT::project(mesh.u(vs[0]),
                             mesh.u(vs[1]),
                             mesh.u(vs[2]),
                             mesh.u(vs[3]),
                             x.template segment<3>(3*mesh.vidx(vs[0])),
                             x.template segment<3>(3*mesh.vidx(vs[1])),
                             x.template segment<3>(3*mesh.vidx(vs[2])),
                             x.template segment<3>(3*mesh.vidx(vs[3])),
                             C, dC[0], dC[1], dC[2], dC[3]);
        }
    };


    template <typename MeshT, typename EnergyT>
    static void project_energy(MeshT& mesh, const typename MeshT::VecXr& x, typename MeshT::Real dt,
                std::vector<int>& constraints_per_vertex,
                typename MeshT::VecXr& lambda,
                typename MeshT::VecXr& dx)
    {
            int idx=0;
            for (auto it=mesh.template begin<typename EnergyT::ElementT>();
                    it!=mesh.template end<typename EnergyT::ElementT>(); ++it, ++idx) {
                Eigen::Matrix<typename MeshT::Real, EnergyT::dim, 1> C;
                std::array<Eigen::Matrix<typename MeshT::Real, 3, EnergyT::dim>, EnergyT::ElementT::size> dC;

                const auto vs = mesh.template vertices<typename EnergyT::ElementT>(it);
                project_n<MeshT, EnergyT, EnergyT::ElementT::size>::_(mesh, x, vs, C, dC);

                const auto a = (1.0 / EnergyT::ks(mesh, it)) / (dt * dt);
                const auto b = dt * dt * EnergyT::kd(mesh, it);
                const auto c = a * b / dt;

                for (int i=0; i<EnergyT::dim; i++) {
                    typename MeshT::Real num = 0.0;
                    typename MeshT::Real den = 0.0;
                    for (int j=0; j<EnergyT::ElementT::size; j++) {
                        num += dC[j].col(i).dot(x.template segment<3>(3*mesh.vidx(vs[j])) - mesh.x(vs[j]));
                        den += dC[j].col(i).squaredNorm() / mesh.mass(vs[j]);
                    }

                    typename MeshT::Real dl = (-C[i] - lambda[idx*EnergyT::dim+i] * a - c * num) / ((1.0 + c) * den + a);
                    lambda[idx*EnergyT::dim+i] += dl;

                    for (int j=0; j<EnergyT::ElementT::size; j++) {
                        dx.template segment<3>(3*mesh.vidx(vs[j])) += dC[j].col(i) * dl / mesh.mass(vs[j]);
                    }
                }

                for (int j=0; j<EnergyT::ElementT::size; j++) {
                    constraints_per_vertex[mesh.vidx(vs[j])]++;
                }
            }
    }

    template <typename MeshT>
    static void project_energies(MeshT&, const typename MeshT::VecXr&, typename MeshT::Real,
            std::vector<int>&,
            std::unordered_map<std::type_index, typename MeshT::VecXr>&,
                 typename MeshT::VecXr&) { }

    template <typename MeshT, typename EnergyT, typename... Energies>
    static void project_energies(MeshT& mesh, const typename MeshT::VecXr& x, typename MeshT::Real dt,
std::vector<int>& constraints_per_vertex,
            std::unordered_map<std::type_index, typename MeshT::VecXr>& lambdas,
                 typename MeshT::VecXr& dx) {

        project_energy<MeshT,EnergyT>(mesh, x, dt, constraints_per_vertex, lambdas.at(std::type_index(typeid(EnergyT))), dx);

        project_energies<MeshT,Energies...>(mesh, x, dt, constraints_per_vertex, lambdas, dx);
    }

    template <typename MeshT, typename AvatarT, typename OptionT, typename... Energies>
    static void step(MeshT& mesh, const AvatarT& avatar, const OptionT& opts)
    {
        const Vec3d gravity(0.0, -980.0, 0.0);

        const double omega = 1.5; // Over-relaxation constant

        static int iteration = 0;
        iteration++;
        std::cout << "Iteration: " << iteration << std::endl;

        const int n_verts = mesh.template n_elements<typename MeshT::Vertex>();
 
        typename MeshT::VecXr x(3*n_verts);
        typename MeshT::VecXr v(3*n_verts);
        for (auto it=mesh.template begin<typename MeshT::Vertex>();
                it != mesh.template end<typename MeshT::Vertex>(); ++it) {
            x.template segment<3>(3*mesh.vidx(it)) = mesh.x(it);
            v.template segment<3>(3*mesh.vidx(it)) = mesh.v(it);
        }

        typename MeshT::VecXr a_ext(3*n_verts);
        if (iteration > 15 || !is_stitching_phase(mesh, 0.1) || !avatar.n_vertices()) {
            for (auto it=mesh.template begin<typename MeshT::Vertex>();
                    it != mesh.template end<typename MeshT::Vertex>(); ++it) {
                a_ext.template segment<3>(3*mesh.vidx(it)) = mesh.is_fixed(it) ? MeshT::Vec3r::Zero() : gravity;
            }
        } else {
            v.setZero();
            a_ext.setZero();
        }

        // Explicit step
        VecXd x_curr = x + v * opts.dt() + a_ext * opts.dt() * opts.dt();

        VecXd dx = VecXd::Zero(x.size());

        const int nbr_iterations = 25;
        int curr_iter = 0;

        /*
        AvatarCollisionsEnergy avatar_force;
        avatar_force.reset();
        auto cs = avatar.n_vertices() ? Collisions::get_avatar_collisions(avatar, x_curr, 0.2) : std::vector<Collision>();

        SelfCollisionForce self_forces;
        std::vector<std::pair<int, int>> vf_pairs;

        const bool do_layer_intersections = true;
        if (do_layer_intersections) {
            mesh.bvh.refit(mesh.f, mesh.x, 2.0);

            mesh.bvh.self_intersect(mesh.f, [&](int v, int f) {
                    if (mesh.f(f,0) == v ||
                        mesh.f(f,1) == v ||
                        mesh.f(f,2) == v) {
                        return;
                    }

                    // Only doing layered collisions for now
                    if (mesh.vl[v] == mesh.fl[f]) {
                        return;
                    }

                    #pragma omp critical
                    { vf_pairs.push_back({ v, f }); }
            });
        }

        self_forces.reset(vf_pairs.size());
        std::cout << "Found " << vf_pairs.size() << " close elements.\n";

        mesh.flags = Eigen::VectorXi::Zero(mesh.x.size() / 3);

        for (const auto& vf : vf_pairs) {
            mesh.flags[vf.first] = 1;
            for (int i=0; i<3; i++) {
                mesh.flags[mesh.f(vf.second,i)] = 1;
            }
        }
        */

        std::vector<int> constraints_per_vertex(n_verts, 0);

        std::unordered_map<std::type_index, typename MeshT::VecXr> lambdas;
        init_lambdas<MeshT,Energies...>(mesh, lambdas);

        while (curr_iter < nbr_iterations) {
            dx.setZero();
            project_energies<MeshT,Energies...>(mesh, x_curr, opts.dt(), constraints_per_vertex, lambdas, dx);

            for (int i=0; i<n_verts; i++) {
                x_curr.template segment<3>(3*i).noalias() += dx.template segment<3>(3*i) * omega / double(constraints_per_vertex[i]);
            }

            /*
            if (cs.size()) {
                dx.setZero();
                avatar_force.update(mesh, x_curr, avatar, cs, 0.2, opts.dt(), dx);
                x_curr.noalias() += dx;
            }

            if (do_layer_intersections) {
                dx.setZero();
                std::vector<int> counts(mesh.x.size()/3, 0);
                self_forces.update(mesh, vf_pairs, counts, x_curr, opts.dt(), 0.25, dx);

                for (int i=0; i<mesh.x.size()/3; i++) {
                    if (counts[i] > 0) {
                        x_curr.template segment<3>(3*i).noalias() += dx.template segment<3>(3*i) * omega / double(counts[i]);
                    }
                }
            }
            */

            curr_iter++;
        }

        //if (avatar.n_vertices()) {
        //    avatar_force.friction(mesh, cs, 0.5, x_curr);
        //}

        v = (x_curr - x) / opts.dt();
        for (auto it=mesh.template begin<typename MeshT::Vertex>();
                it != mesh.template end<typename MeshT::Vertex>(); ++it) {
            mesh.set_x(it, x_curr.template segment<3>(3*mesh.vidx(it)));
            mesh.set_v(it, v.template segment<3>(3*mesh.vidx(it)));
        }

        //mesh.update_normals();
    }
};


