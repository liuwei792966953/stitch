// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

// Use RowMajor so that matrix-vector products are multi-threaded by Eigen
using SparseMatrixd = Eigen::SparseMatrix<double,Eigen::RowMajor>;
//using SparseMatrixd = Eigen::SparseMatrix<double>;


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

