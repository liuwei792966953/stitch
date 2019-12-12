
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

    void precompute(const TriMesh& mesh) override;

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const override;

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd> &triplets) const override;

    size_t nbrEnergies(const TriMesh& mesh) const override { return mesh.e.rows(); }

    void update(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) override;

protected:
    Mat2d clampMatrixEigenvalues(Mat2d &input) const;

protected:
    struct PrecomputedTriangleElement
    {
        double A;
        double edgeLengths[2];
        Mat2d DmInverse;
        Mat2x3d Bm;

        MatXd dFdu;
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    std::vector<PrecomputedTriangleElement> elements_;
    std::vector<double> L_;

    double ksx_;
    double ksy_;
};

