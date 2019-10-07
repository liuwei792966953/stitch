
#pragma once

#include "energy.hpp"
#include <vector>



class NonlinearElasticEnergy : public BaseEnergy
{
public:
    NonlinearElasticEnergy(double ksx, double ksy) : ksx_(ksx), ksy_(ksy) {}
    NonlinearElasticEnergy() = delete;
    ~NonlinearElasticEnergy() = default;

    NonlinearElasticEnergy(const NonlinearElasticEnergy&) = default;
    NonlinearElasticEnergy(NonlinearElasticEnergy&&) = default;
    NonlinearElasticEnergy& operator=(const NonlinearElasticEnergy&) = default;

    void precompute(const TriMesh& mesh);

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const;

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd> &triplets) const;

protected:
    Mat2d clampMatrixEigenvalues(Mat2d &input) const;

protected:
    struct PrecomputedTriangleElement
    {
        Real A;
        Real edgeLengths[2];
        Mat2d DmInverse;
        Mat2x3d Bm;

        MatXd dFdu;
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    std::vector<PrecomputedTriangleElement> elements_;

    double ksx_;
    double ksy_;
};

