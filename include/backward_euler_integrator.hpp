
#pragma once

#include "avatar_collisions.hpp"
#include "collisions.hpp"
#include "conjugate_gradient.hpp"
#include "energy.hpp"
#include <unordered_map>


class BackwardEulerIntegrator {
public:
    std::vector<std::unique_ptr<BaseEnergy>> energies;

    void initialize(const TriMesh& mesh, double dt);

    void step(TriMesh& mesh, double dt, int internal_iters, double kd, double mu);

    void addPin(int idx) { pins_.insert(idx); }

    void addAvatar(AnimatedMesh& mesh) { avatar = &mesh; }

protected:
    bool is_stitching_phase(TriMesh& mesh, double tol=0.5);

protected:
    SparseMatrixd dFdx_;
    SparseMatrixd dFdv_;
    VecXd F_;
    VecXd dx_;
    VecXd dv_;
    VecXd rhs_;
    VecXd resid_;

    std::unordered_set<int> pins_;

    AnimatedMesh* avatar = nullptr;

    AvatarCollisionsEnergy avatar_force_;

    ModifiedConjugateGradient cg_;
};


