// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <Eigen/Core>
#include "BVH.hpp"

struct TriMesh {
    Eigen::MatrixXi f;
    Eigen::MatrixXi ft;
    Eigen::VectorXd x;
    Eigen::VectorXd vt;

    BVH bvh;

    int idx;
};

struct SimMesh : public TriMesh {
    Eigen::VectorXd v;
    Eigen::VectorXd m;
};

