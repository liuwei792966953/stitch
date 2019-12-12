
#pragma once

#include "energy.hpp"



class ImmediateBucklingEnergy : public BaseEnergy
{
public:
    ImmediateBucklingEnergy(double kb) : kb_(kb) {}

    void precompute(const TriMesh& mesh) override;

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const override;

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd> &triplets) const override;

    void perVertexCount(const TriMesh& mesh, std::vector<int>& counts) const override;
   
    void update(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) override;

protected:
    const double kb_ = 1.0;

    std::vector<std::pair<int,int>> pairs_;
    std::vector<double> restLength_;
    std::vector<double> triAreas_;
    std::vector<bool> isCreased_;

    bool across_stitches_ = false;
};


template <typename MeshT>
struct ImmediateBucklingModel
{
    using ElementT = typename MeshT::Diamond;

    static constexpr int dim = 1;

    static int n_constraints() { return 1; }

    static typename MeshT::Real ks(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.bend_ks(it);
    }
    
    static typename MeshT::Real kd(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.bend_kd(it);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    static void project(const Eigen::MatrixBase<DerivedA>& u0, 
                 const Eigen::MatrixBase<DerivedA>& u1,
                 const Eigen::MatrixBase<DerivedA>& u2,
                 const Eigen::MatrixBase<DerivedA>& u3,
                 const Eigen::MatrixBase<DerivedB>& x0,
                 const Eigen::MatrixBase<DerivedB>& x1,
                 const Eigen::MatrixBase<DerivedB>& x3,
                 const Eigen::MatrixBase<DerivedB>& x4,
                 Eigen::MatrixBase<DerivedC>& C,
                 Eigen::MatrixBase<DerivedD>& dC0,
                 Eigen::MatrixBase<DerivedD>& dC1,
                 Eigen::MatrixBase<DerivedD>& dC2,
                 Eigen::MatrixBase<DerivedD>& dC3)
    {
        auto n = x1 - x0;
        
        const auto L = (u1 - u0).norm(); // Rest length
        const auto l = n.norm();         // Current length

        dC0 = -n;
        dC1 =  n;

        // Only resists compression
        C[0] = l < L ? l - L : 0.0;

        if (l > 1.0e-8) {
            dC0 /= l;
            dC1 /= l;
        }
    }

};

/*
template <typename MeshT>
struct IBM
{
    using ElementT = typename MeshT::Edge;

    static constexpr int dim = 1;
    
    static int n_constraints() { return 1; }

    static typename MeshT::Real ks(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.bend_ks(it);
    }
    
    static typename MeshT::Real kd(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.bend_kd(it);
    }

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    static void project(const Eigen::MatrixBase<DerivedA>& u0, 
                 const Eigen::MatrixBase<DerivedA>& u1,
                 const Eigen::MatrixBase<DerivedB>& x0,
                 const Eigen::MatrixBase<DerivedB>& x1,
                 Eigen::MatrixBase<DerivedC>& C,
                 Eigen::MatrixBase<DerivedD>& dC0,
                 Eigen::MatrixBase<DerivedD>& dC1)
    {
        auto n = x1 - x0;
        
        const auto L = (u1 - u0).norm(); // Rest length
        const auto l = n.norm();         // Current length

        dC0 = -n;
        dC1 =  n;

        C[0] = l - L;

        if (l > 1.0e-8) {
            dC0 /= l;
            dC1 /= l;
        }
    }
};
*/
