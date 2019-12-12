// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "mesh.hpp"

// Use RowMajor so that matrix-vector products are multi-threaded by Eigen
using SparseMatrixd = Eigen::SparseMatrix<double,Eigen::RowMajor>;
//using SparseMatrixd = Eigen::SparseMatrix<double>;

using SparseTripletd = Eigen::Triplet<double>;

using VecXd = Eigen::VectorXd;
using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Mat2d = Eigen::Matrix2d;
using Mat3d = Eigen::Matrix3d;
using MatXd = Eigen::MatrixXd;
using Mat2x3d = Eigen::Matrix<double, 2, 3>;
using Mat3x2d = Eigen::Matrix<double, 3, 2>;


// Abstract interface to define an energy term
class Energy {
public:
    void set_index(int i) { index_ = i; }
    int index() const { return index_; }

    const Eigen::VectorXd& weights() const { return weights_; };
    
    virtual int dim() const = 0;

    virtual Eigen::VectorXd reduce(const Eigen::VectorXd& x) const = 0;

    virtual void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const = 0;
    virtual void project(Eigen::VectorXd &zi) const = 0;

    virtual void update(int iter) { }

protected:
    Eigen::VectorXd weights_;

private:
    int index_;
};

class DynamicEnergy : public Energy {
public:
    virtual void multiply(const Eigen::VectorXd& x,
                          const Eigen::VectorXd& factor,
                          const Eigen::VectorXd& shift,
                          Eigen::VectorXd& out) const = 0;
};


class BaseEnergy
{
public:
    virtual ~BaseEnergy() { }

    virtual void precompute(const TriMesh& mesh) = 0;

    virtual void getForceAndHessian(const TriMesh& mesh,
                                    const Eigen::VectorXd& x,
				    Eigen::VectorXd& F,
				    SparseMatrixd& dFdx,
				    SparseMatrixd& dFdv) const = 0;

    virtual void getHessianPattern(const TriMesh& mesh,
                                   std::vector<SparseTripletd> &triplets) const = 0;


    // XPBD
    virtual size_t nbrEnergies(const TriMesh& mesh) const { return 0; }

    virtual void perVertexCount(const TriMesh& mesh, std::vector<int>& counts) const { }

    virtual void update(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) { }
    
    void reset() { lambda_.setZero(); }

protected:
    VecXd lambda_; // For XPBD
};
