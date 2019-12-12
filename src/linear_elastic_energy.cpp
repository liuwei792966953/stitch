
#include "linear_elastic_energy.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;



void LinearElasticEnergy::precompute(const TriMesh& mesh)
{
    A_.resize(mesh.f.rows());
    DmInverse_.resize(mesh.f.rows());

    for (int idx=0; idx<mesh.f.rows(); idx++) {
        int i0 = mesh.f(idx, 0);
        int i1 = mesh.f(idx, 1);
        int i2 = mesh.f(idx, 2);

        Mat2d Dm;
        Dm.col(0) = mesh.u.segment<2>(2*i1) - mesh.u.segment<2>(2*i0);
        Dm.col(1) = mesh.u.segment<2>(2*i2) - mesh.u.segment<2>(2*i0);

        DmInverse_[idx] = Dm.inverse();

        A_[idx] = 0.5 * (Vec3d() << Dm.col(0), 0.0).finished().cross(
                        (Vec3d() << Dm.col(1), 0.0).finished()).norm();
    }

    lambda_ = VecXd::Zero(3*mesh.f.rows());
}

void LinearElasticEnergy::getForceAndHessian(const TriMesh& mesh,
                                                const VecXd& x,
                                                VecXd& F,
                                                SparseMatrixd& dFdx,
                                                SparseMatrixd& dFdv) const {
    assert(F.size() >= x.size());

    for (int idx=0; idx<mesh.f.rows(); idx++) {
    }
}

void LinearElasticEnergy::getHessianPattern(const TriMesh& mesh, vector<SparseTripletd> &triplets) const {
    for (int idx=0; idx<mesh.f.rows(); idx++) {
        int idxs[3] = { mesh.f(idx,0), mesh.f(idx,1), mesh.f(idx,2) };
        for (size_t j=0; j<3; ++j)
            for (size_t k=0; k<3; ++k)
                for (size_t l=0; l<3; ++l)
                    for (size_t n=0; n<3; ++n)
                        triplets.push_back(SparseTripletd(3*idxs[j]+l,3*idxs[k]+n, 1.0));
    }
}

void LinearElasticEnergy::perVertexCount(const TriMesh& mesh, std::vector<int>& counts) const {
    for (int idx=0; idx<mesh.f.rows(); idx++) {
        for (int j=0; j<3; j++) {
            counts[mesh.f(idx,j)]++;
        }
    }
}

void LinearElasticEnergy::update(const TriMesh& mesh, const VecXd& x, double dt, VecXd& dx) {
    const double a = (1.0 / ksx_) / (dt * dt);

    for (int idx=0; idx<mesh.f.rows(); idx++) {
        const int idxs[3] = { mesh.f(idx,0), mesh.f(idx,1), mesh.f(idx,2) };

        Mat3x2d Dw;
        for (int i=0; i<2; i++) {
            Dw.col(i) = x.segment<3>(3*idxs[i+1]) - x.segment<3>(3*idxs[0]);// +
                            //dx.segment<3>(3*idxs[i+1]) - dx.segment<3>(3*idxs[0]); // Jacobi
        }

        // Deformation gradient
        Mat3x2d F = Dw * DmInverse_[idx];

        const double wu_mag = F.col(0).norm();
        const double wv_mag = F.col(1).norm();

        Vec3d wu = F.col(0) / wu_mag;
        Vec3d wv = F.col(1) / wv_mag;

        // C(x) = A ( || wu || - bu  || wv || - bv    wu . wv )
        Vec3d C = A_[idx] * Vec3d(wu_mag - 1.0, wv_mag - 1.0, F.col(0).dot(F.col(1)));

        // grad C is 3, 3x3 matrices, where each is w.r.t a different vertex
        // Each column is the derivative of the respective C
        Mat3d gradC[3];

        gradC[1].col(0) = A_[idx] * DmInverse_[idx](0,0) * wu;
        gradC[1].col(1) = A_[idx] * DmInverse_[idx](0,1) * wv;
        gradC[1].col(2) = A_[idx] * (DmInverse_[idx](0,0) * F.col(1) + DmInverse_[idx](0,1) * F.col(0));
        gradC[2].col(0) = A_[idx] * DmInverse_[idx](1,0) * wu;
        gradC[2].col(1) = A_[idx] * DmInverse_[idx](1,1) * wv;
        gradC[2].col(2) = A_[idx] * (DmInverse_[idx](1,0) * F.col(1) + DmInverse_[idx](1,1) * F.col(0));
        gradC[0] = -(gradC[1] + gradC[2]);

	double den1 = (gradC[0].col(0).squaredNorm() / mesh.m[3*idxs[0]] +
	               gradC[1].col(0).squaredNorm() / mesh.m[3*idxs[1]] +
	               gradC[2].col(0).squaredNorm() / mesh.m[3*idxs[2]] + a);
	double den2 = (gradC[0].col(1).squaredNorm() / mesh.m[3*idxs[0]] +
	               gradC[1].col(1).squaredNorm() / mesh.m[3*idxs[1]] +
	               gradC[2].col(1).squaredNorm() / mesh.m[3*idxs[2]] + a);
	double den3 = (gradC[0].col(2).squaredNorm() / mesh.m[3*idxs[0]] +
	               gradC[1].col(2).squaredNorm() / mesh.m[3*idxs[1]] +
	               gradC[2].col(2).squaredNorm() / mesh.m[3*idxs[2]] + a);

	Vec3d dl = (-C - lambda_.segment<3>(3*idx) * a).array() / Eigen::Array3d(den1, den2, den3);

	dx.segment<3>(3*idxs[0]) += gradC[0] * dl / mesh.m[3*idxs[0]];
	dx.segment<3>(3*idxs[1]) += gradC[1] * dl / mesh.m[3*idxs[1]];
	dx.segment<3>(3*idxs[2]) += gradC[2] * dl / mesh.m[3*idxs[2]];

	lambda_.segment<3>(3*idx) += dl;
    }
}
