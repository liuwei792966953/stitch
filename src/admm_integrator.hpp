// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <igl/barycentric_coordinates.h>
#include <unordered_set>

#include "conjugate_gradient.hpp"
#include "energy.hpp"
#include "mesh.hpp"



class ADMM_Integrator {
public:
    std::vector<std::shared_ptr<Energy>> energies;

    void initialize(const SimMesh& mesh, double dt);

    struct Collision {
        int tri_idx = -1;
        double dx = 0.0;
        Eigen::Vector3d n;
    };

    void step(SimMesh& mesh, double dt, int internal_iters, double kd, double mu);

    void addPin(int idx) { pins_.insert(idx); }

    void addAvatar(TriMesh& mesh) { avatar = &mesh; }

protected:
    Eigen::VectorXd z_;
    Eigen::VectorXd u_;
    Eigen::VectorXd dx_;

    SparseMatrixd A_;
    SparseMatrixd D_;
    SparseMatrixd DtWtW_;

    std::unordered_set<int> pins_;

    std::vector<Collision> collisions_;

    TriMesh* avatar = nullptr;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
    ModifiedConjugateGradient cg_;
};


