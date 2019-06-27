// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

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
                        //if (dist > offset) {
                        //    cs[i].dx = dist;
                        //    cs[i].tri_idx = -1;
                        //}
                        if (dist < cs[i].dx) {// && dist > -0.5 && cs[i].dx < 1.1 * offset) {
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

}
