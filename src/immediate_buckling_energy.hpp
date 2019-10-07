
#pragma once

#include "energy.hpp"



class ImmediateBucklingEnergy : public BaseEnergy
{
public:
    ImmediateBucklingEnergy(double kb) : kb_(kb) {}

    void precompute(const TriMesh& mesh);

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const;

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd> &triplets) const;
   
protected:
    const double kb_ = 1.0;

    std::vector<std::pair<int,int>> pairs_;
    std::vector<Real> restLength_;
    std::vector<Real> triAreas_;
    std::vector<bool> isCreased_;

    bool across_stitches_ = false;
};

