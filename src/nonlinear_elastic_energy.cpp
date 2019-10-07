
#include "nonlinear_elastic_energy.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;



void NonlinearElasticEnergy::precompute(const TriMesh& mesh)
{
    elements_.resize(mesh.f.rows());

    Mat3x2d selector;
    selector << 0.0, 0.0,
                1.0, 0.0,
                0.0, 1.0;

    for (int idx=0; idx<mesh.f.rows(); idx++) {
        int i0 = mesh.f(idx, 0);
        int i1 = mesh.f(idx, 1);
        int i2 = mesh.f(idx, 2);

        Mat2d Dm;
        Dm.col(0) = mesh.u.segment<2>(2*i1) - mesh.u.segment<2>(2*i0);
        Dm.col(1) = mesh.u.segment<2>(2*i2) - mesh.u.segment<2>(2*i0);

        const Real A = 0.5 * (Vec3d() << Dm.col(0), 0.0).finished().cross(
                             (Vec3d() << Dm.col(1), 0.0).finished()).norm();

        elements_[idx].A = A;

        for (size_t i=0; i<2; i++)
            elements_[idx].edgeLengths[i] = Dm.col(i).norm();

        Mat3d P;
        P <<   1.0,   1.0,   1.0,
            mesh.u[2*i0], mesh.u[2*i1], mesh.u[2*i2],
            mesh.u[2*i0+1], mesh.u[2*i1+1], mesh.u[2*i2+1];

        // 2x3 matrix containing the gradients of the
        // 3 shape functions (one for each node)
        elements_[idx].Bm = A * (P.inverse() * selector).transpose();

        elements_[idx].DmInverse = Dm.inverse();

        Mat2d DmI = elements_[idx].DmInverse;
        elements_[idx].dFdu = MatrixXd::Zero(6,9);
        elements_[idx].dFdu << -DmI(0,0)-DmI(1,0), 0.0, 0.0, DmI(0,0), 0.0, 0.0, DmI(1,0), 0.0, 0.0,
                               -DmI(0,1)-DmI(1,1), 0.0, 0.0, DmI(0,1), 0.0, 0.0, DmI(1,1), 0.0, 0.0,
                                0.0, -DmI(0,0)-DmI(1,0), 0.0, 0.0, DmI(0,0), 0.0, 0.0, DmI(1,0), 0.0,
                                0.0, -DmI(0,1)-DmI(1,1), 0.0, 0.0, DmI(0,1), 0.0, 0.0, DmI(1,1), 0.0,
                                0.0, 0.0, -DmI(0,0)-DmI(1,0), 0.0, 0.0, DmI(0,0), 0.0, 0.0, DmI(1,0),
                                0.0, 0.0, -DmI(0,1)-DmI(1,1), 0.0, 0.0, DmI(0,1), 0.0, 0.0, DmI(1,1);

    }
}

void NonlinearElasticEnergy::getForceAndHessian(const TriMesh& mesh,
                                                const VecXd& x,
                                                VecXd& F,
                                                SparseMatrixd& dFdx,
                                                SparseMatrixd& dFdv) const {
    assert(F.size() >= x.size());

    for (int idx=0; idx<mesh.f.rows(); idx++) {
        // We are assuming that poisson ratio is 0.0, and thus, lambda = 0.0
        // Convert (E,v) (Young's modulus and poisson ratio) to lame parameters
        Real mu = ksx_ * 0.5;

        int idxs[3] = { mesh.f(idx,0), mesh.f(idx,1), mesh.f(idx,2) };

        Mat3x2d Dw;
        for (int i=0; i<2; i++) {
            Dw.col(i) = x.segment<3>(3*idxs[i+1]) - x.segment<3>(3*idxs[0]);
        }

        Vec2d ls = Dw.colwise().norm();
        for (int i=0; i<2; i++) {
            if (ls[i] > 2.5 * elements_[idx].edgeLengths[i]) {
                Dw.col(i) *= 2.5 * elements_[idx].edgeLengths[i] / ls[i];
            }
        }

        // Compute the deformation gradient Fdg,
        // 1st and 2nd Piola-Kirchoff stress tensors (S and P)
        Mat3x2d Fdg = Dw * elements_[idx].DmInverse;
	Mat2d     S = mu * (Fdg.transpose() * Fdg - Mat2d::Identity());
        Mat3x2d   P = Fdg * S;

        // Force is the 2nd P-K stress projected onto the node
        for (int i=0; i<3; i++) {
            F.segment<3>(3*idxs[i]).noalias() -= P * elements_[idx].Bm.col(i);
        }

        S = clampMatrixEigenvalues(S);

        // Derivative of 2nd P-K stress tensor w.r.t. F, the deformation gradient
        //
        // \partial S11 / \partial F = 4 mu ( F.col(0)  0 )
        // \partial S22 / \partial F = 4 mu ( 0  F.col(1) )
        // \partial S12 / \partial F = \partial S21 / \partial F = ( F.col(1)  F.col(0) )
        Mat3x2d dSdF[4];
        dSdF[0].col(0) = 2.0 * mu * Fdg.col(0);
        dSdF[0].col(1) = dSdF[3].col(0) = Vec3d::Zero();
        dSdF[3].col(1) = 2.0 * mu * Fdg.col(1);
        dSdF[1].col(0) = dSdF[2].col(0) = mu * Fdg.col(1);
        dSdF[1].col(1) = dSdF[2].col(1) = mu * Fdg.col(0);

        const Mat2x3d& dN = elements_[idx].Bm;

        // TODO: Work out compact expressions that take advantage of dFdu_jk only having one
        // non-zero row. The row is also a function of DmInverse above. No need to pre-compute more.
        Vec3d fi_jk;
        Mat2d dSdu_jk;
        for (size_t i=0; i<3; ++i)
        {
            for (size_t j=0; j<3; ++j)
            {
                for (size_t k=0; k<3; ++k) // \partial f_i / \partial u_j^k
                {
                    // 3 x 2 matrix that is the derivative of F with respect to u_j^k
                    //
                    Mat3x2d dFdu_jk = elements_[idx].dFdu.block<2,3>(2*k, 3*j).transpose();

                    for (size_t l=0; l<2; ++l)
                        for (size_t m=0; m<2; ++m)
                            dSdu_jk(l,m) = (dSdF[2*l+m].array() * dFdu_jk.array()).sum();

                    // Force on vertex i is f_i = F P dN_i, so
                    // derivative w.r.t. DOF u_j^k (j-th vertex, k-th DOF) is:
                    //
                    // \partial f_i / \partial u_j^k =
                    //     ( \partial F / \partial u_j^k S + F \partial S / \partial u_j^k ) dN_i
                    fi_jk = (dFdu_jk * S + Fdg * dSdu_jk) * dN.col(i);

                    for (size_t l=0; l<3; ++l)
                        //J.coeffRef(3*idxs[j]+k, 3*idxs[i]+l) += fi_jk[l];
                        dFdx.coeffRef(3*idxs[i]+l, 3*idxs[j]+k) -= fi_jk[l];
                }
            }
        }
    }
}

Mat2d NonlinearElasticEnergy::clampMatrixEigenvalues(Mat2d &input) const {
    static const Real eps = 1.0e-6;

    SelfAdjointEigenSolver<Mat2d> ev(input);
    assert(ev.info() == Success);

    return (ev.eigenvectors() *
            ev.eigenvalues().cwiseMax(eps).asDiagonal() *
            ev.eigenvectors().transpose());
}

void NonlinearElasticEnergy::getHessianPattern(const TriMesh& mesh, vector<SparseTripletd> &triplets) const {
    for (int idx=0; idx<mesh.f.rows(); idx++) {
        int idxs[3] = { mesh.f(idx,0), mesh.f(idx,1), mesh.f(idx,2) };
        for (size_t j=0; j<3; ++j)
            for (size_t k=0; k<3; ++k)
                for (size_t l=0; l<3; ++l)
                    for (size_t n=0; n<3; ++n)
                        triplets.push_back(SparseTripletd(3*idxs[j]+l,3*idxs[k]+n, 1.0));
    }
}

