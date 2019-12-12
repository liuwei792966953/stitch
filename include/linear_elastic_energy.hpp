
#pragma once

#include "energy.hpp"
#include <vector>



class LinearElasticEnergy : public BaseEnergy
{
public:
    LinearElasticEnergy(double ksx, double ksy) : ksx_(ksx), ksy_(ksy) {}
    LinearElasticEnergy() = delete;
    ~LinearElasticEnergy() = default;

    LinearElasticEnergy(const LinearElasticEnergy&) = default;
    LinearElasticEnergy(LinearElasticEnergy&&) = default;
    LinearElasticEnergy& operator=(const LinearElasticEnergy&) = default;

    void precompute(const TriMesh& mesh) override;

    void getForceAndHessian(const TriMesh& mesh, const VecXd& x,
				VecXd& F,
				SparseMatrixd& dFdx,
				SparseMatrixd& dFdv) const override;

    void getHessianPattern(const TriMesh& mesh, std::vector<SparseTripletd> &triplets) const override;

    size_t nbrEnergies(const TriMesh& mesh) const override { return mesh.e.rows(); }

    void update(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) override;

    void perVertexCount(const TriMesh& mesh, std::vector<int>& counts) const override;

protected:
    std::vector<double> A_;
    std::vector<Mat2d> DmInverse_;

    double ksx_;
    double ksy_;
};


template <typename MeshT>
struct LinearElasticity
{
    using ElementT = typename MeshT::Triangle;

    static constexpr int dim = 3;

    static int n_constraints() { return 3; }

    static typename MeshT::Real ks(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.stretch_ks(it);
    }
    
    static typename MeshT::Real kd(const MeshT& mesh, typename ElementT::iterator it) {
        return mesh.stretch_kd(it);
    }

    using Mat2r = Eigen::Matrix<typename MeshT::Real, 2, 2>;
    using Mat3x2r = Eigen::Matrix<typename MeshT::Real, 3, 2>;

    template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    static void project(const Eigen::MatrixBase<DerivedA>& u0, 
                 const Eigen::MatrixBase<DerivedA>& u1,
                 const Eigen::MatrixBase<DerivedA>& u2,
                 const Eigen::MatrixBase<DerivedB>& x0,
                 const Eigen::MatrixBase<DerivedB>& x1,
                 const Eigen::MatrixBase<DerivedB>& x2,
                 Eigen::MatrixBase<DerivedC>& C,
                 Eigen::MatrixBase<DerivedD>& dC0,
                 Eigen::MatrixBase<DerivedD>& dC1,
                 Eigen::MatrixBase<DerivedD>& dC2)
    {
        Mat2r Dm;
        Dm.col(0) = (u1 - u0).template head<2>();
        Dm.col(1) = (u2 - u0).template head<2>();

        Mat2r DmInverse = Dm.inverse();

        Mat3x2r Dw;
        Dw.col(0) = x1 - x0;
        Dw.col(1) = x2 - x0;

        Mat3x2r F = Dw * DmInverse;

        auto A = 0.5 * Dm.determinant();

        const typename MeshT::Real wu_mag = F.col(0).norm();
        const typename MeshT::Real wv_mag = F.col(1).norm();

        typename MeshT::Vec3r wu = F.col(0) / wu_mag;
        typename MeshT::Vec3r wv = F.col(1) / wv_mag;

        // C(x) = A ( || wu || - bu  || wv || - bv    wu . wv )
        C = A * typename MeshT::Vec3r(wu_mag - 1.0, wv_mag - 1.0, F.col(0).dot(F.col(1)));

        dC1.col(0) = A * DmInverse(0,0) * wu;
        dC1.col(1) = A * DmInverse(0,1) * wv;
        dC1.col(2) = A * (DmInverse(0,0) * F.col(1) + DmInverse(0,1) * F.col(0));
        dC2.col(0) = A * DmInverse(1,0) * wu;
        dC2.col(1) = A * DmInverse(1,1) * wv;
        dC2.col(2) = A * (DmInverse(1,0) * F.col(1) + DmInverse(1,1) * F.col(0));
        dC0 = -(dC1 + dC2);
    }
};
