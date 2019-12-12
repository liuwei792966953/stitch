// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <igl/barycentric_coordinates.h>

#include "bvh.hpp"
#include "mesh.hpp"


struct Collision {
    int tri_idx = -1;
    double dx = 0.0;
    Eigen::Vector3d n;
    Eigen::Vector3d w;
    Eigen::Vector3d av_dx = Eigen::Vector3d::Zero();
};

namespace Collisions {

    inline
    std::vector<Collision> get_avatar_collisions(const AnimatedMesh& avatar,
                                  const Eigen::VectorXd& x,
                                  double offset) {
        std::vector<Collision> cs(x.size() / 3);

        #pragma omp parallel for
        for (int i=0; i<x.size() / 3; i++) {
            cs[i].dx = offset;

            avatar.bvh.visit(x.segment<3>(3*i),
                    [&](int idx) {
                    Eigen::RowVector3d w;
                    igl::barycentric_coordinates(x.segment<3>(3*i).transpose(),
                            avatar.x.segment<3>(3*avatar.f(idx,0)).transpose(),
                            avatar.x.segment<3>(3*avatar.f(idx,1)).transpose(),
                            avatar.x.segment<3>(3*avatar.f(idx,2)).transpose(), w);

                    if (std::all_of(w.data(), w.data()+3,
                                [](double v) { return v > -0.001 && v < 1.001; })) {

                        Eigen::Vector3d n = (avatar.x.segment<3>(3*avatar.f(idx,1)) -
                                             avatar.x.segment<3>(3*avatar.f(idx,0))).cross(
                                             avatar.x.segment<3>(3*avatar.f(idx,2)) -
                                             avatar.x.segment<3>(3*avatar.f(idx,0))).normalized();

                        double dist = (x.segment<3>(3*i) - avatar.x.segment<3>(3*avatar.f(idx,0))).dot(n);

                        if (dist < cs[i].dx && dist > -2.5) {
                            cs[i].dx = dist;
                            cs[i].tri_idx = idx;
                            cs[i].n = n;
                            cs[i].w = w.transpose();
                        }
                    }
            });
        }

        return cs;
    }

    inline
    bool get_barycentric_weights(const Eigen::Vector3d& p,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c,
                                 Eigen::Vector3d& w) {
        const Eigen::Vector3d v0 = b - a;
        const Eigen::Vector3d v1 = c - a;
        const Eigen::Vector3d v2 = p - a;

        const double d00 = v0.dot(v0);
        const double d01 = v0.dot(v1);
        const double d11 = v1.dot(v1);
        const double d20 = v2.dot(v0);
        const double d21 = v2.dot(v1);
        const double denom = d00 * d11 - d01 * d01;
        if (std::fabs(denom) < 1.0e-8) {
            w[0] = w[1] = w[2] = 1. / 3.;
            return false;
        }

        w[1] = (d11 * d20 - d01 * d21) / denom;
        w[2] = (d00 * d21 - d01 * d20) / denom;
        w[0] = 1.0 - w[1] - w[2];

        if (std::any_of(w.data(), w.data() + 3,
                    [](const double v) { return v < -0.001 || v > 1.001; })) {
            return false;
        }

        for (int i = 0; i < 3; i++) {
            w[i] = std::max(std::min(w[i], 1.0), 0.0);
        }

        return true;
    }

}
