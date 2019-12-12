#pragma once

#include <unordered_map>

#include "bvh.hpp"
#include "mesh.hpp"


class Constraint {
public:
    virtual void detect(TriMesh& mesh, std::unordered_map<int, Eigen::Matrix3d>& C) = 0;
};


class AvatarConstraint {
public:
    AvatarConstraint(TriMesh& avatar) : avatar_(avatar) {}

    void detect(TriMesh& mesh, std::unordered_map<int, Eigen::Matrix3d>& C) {
    }

protected:
    TriMesh& avatar_;

    std::unordered_set<int> collisions_;
};

