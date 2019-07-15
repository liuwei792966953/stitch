// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <igl/barycentric_coordinates.h>
#include <unordered_set>

#include "collisions.hpp"
#include "conjugate_gradient.hpp"
#include "energy.hpp"
#include "gauss_seidel.hpp"
#include "jacobi.hpp"
#include "mesh.hpp"
#include "successive_over_relaxation.hpp"



class ADMM_Integrator {
public:
    std::vector<std::shared_ptr<Energy>> energies;

    void initialize(const TriMesh& mesh, double dt);

    void step(TriMesh& mesh, double dt, int internal_iters, double kd, double mu);

    void addPin(int idx) { pins_.insert(idx); }

    void addAvatar(AnimatedMesh& mesh) { avatar = &mesh; }

protected:
    Eigen::VectorXd z_;
    Eigen::VectorXd u_;
    Eigen::VectorXd dx_;

    SparseMatrixd A_;
    SparseMatrixd DtWtW_;

    std::unordered_set<int> pins_;

    std::vector<Collision> collisions_;

    AnimatedMesh* avatar = nullptr;

    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
    //ConstrainedJacobi cg_;
    //ConstrainedGaussSeidel cg_;
    //ConstrainedSSOR cg_;
    ModifiedConjugateGradient cg_;
};


