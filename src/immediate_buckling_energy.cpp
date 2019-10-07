
#include "immediate_buckling_energy.hpp"

using namespace std;
using namespace Eigen;


/** Bending hinge stencil: only i0 and i1 are affected with the IBM force
 *
 *     i0
 *     /\
 *    /  \
 *  i2----i3
 *    \  /
 *     \/
 *     i1
 */


// The function f_b and its derivative, from the paper. Slight difference: these are
// a function of lambda = l / L, or | x_ij | / L as it is called in the paper. Factoring
// it this way requires an extra coefficient down below, but means one hard-coded polynomial
// can represent all possible hinges.
//
Real fb_poly(const Real x)
{
  Real x2 = x * x;
  Real x3 = x2 * x;
  Real x4 = x3 * x;
  Real x5 = x4 * x;
  return -3.32038180709 - 0.000606180655699 * x - 0.108443942579 * x2 - 0.0202089099968 * x3 + 0.00808034177691 * x4 - 0.0237821521801 * x5;
  //return (-11.541 * x4 + 34.193 * x3 - 39.083 * x2 + 23.116 * x - 9.713);
}

Real dfb_poly(const Real x)
{
  Real x2 = x * x;
  Real x3 = x2 * x;
  Real x4 = x3 * x;
  return -0.000606180655699 - 2.0f * 0.108443942579 * x - 3.0f * 0.0202089099968 * x2 + 4.0f * 0.00808034177691 * x3 - 5.0f * 0.0237821521801 * x4;
  //return (4.0 * (-11.541)) * x3 + (3.0 * 34.193) * x2  - (2.0 * 39.083) * x + 23.116;
}

// Align xs[4] - xs[3] to xs[1] - xs[0] and return in new space
inline
std::array<Eigen::Vector3d, 4> get_aligned(const std::array<Eigen::Vector3d, 6>& xs) {
    Eigen::Vector3d e1 = xs[1] - xs[0];
    Eigen::Vector3d e2 = xs[4] - xs[3];

    Eigen::Vector3d e3 = xs[5] - xs[3];

    if ((e1 - e2).squaredNorm() > 1.0e-8) {
        Eigen::Quaterniond q;
        q.FromTwoVectors(e2, e1);

        // Now rotate other vector by the same amount
        e3 = q * e3;
    }

    return { xs[0], xs[1], xs[2], xs[0] + e3 };
}




void ImmediateBucklingEnergy::precompute(const TriMesh& mesh) {
    std::array<int,4>   idxs;
    std::array<Vec3d,4> xs;

    for (int i=0; i<mesh.ei.rows(); i++) {
        if (mesh.ei(i,0) != -1 && mesh.ei(i,1) != -1) {
            idxs[0] = mesh.e(i,0);
            idxs[1] = mesh.e(i,1);
            idxs[2] = mesh.f(mesh.ef(i, 0), mesh.ei(i, 0));
            idxs[3] = mesh.f(mesh.ef(i, 1), mesh.ei(i, 1));

            for (int j=0; j<4; j++) {
                xs[j] << mesh.u.segment<2>(2*idxs[j]), 0.0;
            }

            Real A1 = 0.5 * (xs[2] - xs[0]).cross(xs[1] - xs[0]).norm();
            Real A2 = 0.5 * (xs[1] - xs[0]).cross(xs[3] - xs[0]).norm();

            triAreas_.push_back(A1 + A2);
            restLength_.push_back((xs[3] - xs[2]).norm());
            pairs_.push_back({ idxs[2], idxs[3] });
        }
    }

    std::array<Vec3d,6> xs_unaligned;
    std::array<Vec3d,4> xs_aligned;
    if (across_stitches_) {
        for (int i=0; i<mesh.s.rows() - 1; i++) {
            const int v0 = mesh.s(i  ,0);
            const int v1 = mesh.s(i+1,0);

            const int e0 = mesh.edge_index(v0, v1);
            if (e0 == -1) {
                continue;
            }

            const int w0 = mesh.s(i  ,1);
            const int w1 = mesh.s(i+1,1);
            const int e1 = mesh.edge_index(w0, w1);
            if (e1 == -1) {
                continue;
            }

            const int o0 = mesh.f(mesh.ef(e0, 0), mesh.ei(e0, 0));
            const int o1 = mesh.f(mesh.ef(e1, 0), mesh.ei(e1, 0));

            if (o0 == -1 || o1 == -1) {
                std::cout << "Opps: " << mesh.ei(e0,1) << "; " << mesh.ei(e1,1) << std::endl;
                continue;
            }

            // The face connected to these two edges
            const int ef0 = mesh.ef(e0);
            const int ef1 = mesh.ef(e1);
            if (ef0 == -1 || ef1 == -1) {
                continue;
            }

            // The stitch goes in the order v0, v1 and w0, w1. Does this
            // order agree with the winding order of the connected faces?
            const bool reversed_stitched0 = mesh.f(ef0, (o0 + 1) % 3) == v0;
            const bool reversed_stitched1 = mesh.f(ef1, (o1 + 1) % 3) == w0;

            // If the winding orders agree, then that means a piece is
            // placed on top of another to create this stitch, and we
            // _don't_ want to create a bending force
            if (reversed_stitched0 == reversed_stitched1) {
                continue;
            }

            // 6 positions, e00, e01, o0, e10, e11, o1

            if (mesh.has_uvs()) {
                Vec2d uv = mesh.vt.segment<2>(2*mesh.uv_index(v0));
                xs_unaligned[0] = (Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(v1));
                xs_unaligned[1] = (Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(o0));
                xs_unaligned[2] = (Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(w0));
                xs_unaligned[3] = (Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(w1));
                xs_unaligned[4] = (Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(o1));
                xs_unaligned[5] = (Eigen::Vector3d(uv[0], uv[1], 0.0));
            } else {
                xs_unaligned[0] = (mesh.x.segment<3>(3*v0));
                xs_unaligned[1] = (mesh.x.segment<3>(3*v1));
                xs_unaligned[2] = (mesh.x.segment<3>(3*o0));
                xs_unaligned[3] = (mesh.x.segment<3>(3*w0));
                xs_unaligned[4] = (mesh.x.segment<3>(3*w1));
                xs_unaligned[5] = (mesh.x.segment<3>(3*o1));
            }

            xs_aligned = get_aligned(xs_unaligned);

            Real A1 = 0.5 * (xs_aligned[2] - xs_aligned[0]).cross(xs_aligned[1] - xs_aligned[0]).norm();
            Real A2 = 0.5 * (xs_aligned[1] - xs_aligned[0]).cross(xs_aligned[3] - xs_aligned[0]).norm();

            triAreas_.push_back(A1 + A2);
            restLength_.push_back((xs_aligned[3] - xs_aligned[2]).norm());
            pairs_.push_back({ o0, o1 });
        }
    }

    std::cout << "Created " << pairs_.size() << " bending forces. " << triAreas_.size() << "; " << restLength_.size() << std::endl;
}

void ImmediateBucklingEnergy::getForceAndHessian(const TriMesh& mesh,
                                                 const VecXd& x,
                                                 VecXd& F,
                                                 SparseMatrixd& dFdx,
                                                 SparseMatrixd& dFdv) const {

    for (size_t idx=0; idx<pairs_.size(); idx++) {
        int i0 = pairs_[idx].first;
        int i1 = pairs_[idx].second;

        if (i0 == -1 || i1 == -1) {
            continue;
        }

        Vec3d x_ij = x.segment<3>(3*i1) - x.segment<3>(3*i0);
        Real curr_length = x_ij.norm();

        // If it is elongated, we let the stretching force handle it, unless it's creased. If
        // the edge is degenerate we also skip it, as dividing by curr_length will blow up otherwise
        if (curr_length > restLength_[idx] || curr_length < 1.0e-8) {
            continue;
        }

        //Vec3d f = x_ij * kb_ * ((curr_length - restLength_[idx]) / curr_length);
        //Mat3d K = (x_ij * x_ij.transpose()) * kb_ / x_ij.dot(x_ij);

        // The direction of the force is always the same, below we compute the magnitude
        Vec3d f = triAreas_[idx] * x_ij / curr_length;
        Mat3d K = triAreas_[idx] * (x_ij * x_ij.transpose()) / x_ij.dot(x_ij);

        // The strain ratio: > 1 indicates elongation, < 1, compression
        const Real lambda = curr_length / restLength_[idx];

        // False unless an elongated creased edge
        if (lambda > 1.0)
        {
            std::cout << "SHOULDN'T BE HERE!" << std::endl;
            assert(0);
            // The stencil is stretched, so we will apply an attractive spring force to pull
            // them together. k_s is the strength of that spring
            Real k_s = 10000.0;

            f *= k_s * (curr_length - restLength_[idx]);
            K *= k_s * restLength_[idx] / curr_length;

            // The stiffness has an additional component (see paper for details) 
            //
            K.noalias() +=
                triAreas_[idx] * k_s * (1.0 - restLength_[idx] / curr_length) * Mat3d::Identity();
        }
        else
        {
            // coeff is the part factored out to make fb a function of lambda
            const Real coeff = 2.0 / restLength_[idx];
            const Real f_mag = coeff * coeff * fb_poly(lambda);
            const Real k_mag = coeff * coeff * dfb_poly(lambda);

            // Arbitrary, to handle the 'buckling instability'
            // TODO: Make it related to stretching stiffness?
            //const Real cb = 1.0e6;
            const Real cb = 20.0;

            // Has the material here "buckled" yet?
            if (kb_ * f_mag < cb * (curr_length - restLength_[idx]))
            {
                f *= cb * (curr_length - restLength_[idx]);
                K *= cb;
            }
            else
            {
                // kb is the bending resistance strength
                f *= kb_ * f_mag;
                K *= kb_ * k_mag;
            }
        }

        const double kd = 1.0;
        Vec3d v_ij = mesh.v.segment<3>(3*i1) - mesh.v.segment<3>(3*i0);

        F.segment<3>(3*i0) += f + v_ij * kd;
        F.segment<3>(3*i1) -= f + v_ij * kd;

        Real signs[2] = { -1.0, 1.0 };
        int idxs[2] = { i0, i1 };
        for (size_t j=0; j<2; ++j) {
            for (size_t k=0; k<2; ++k) {
                for (size_t l=0; l<3; ++l) {
                    for (size_t n=0; n<3; ++n) {
                        dFdx.coeffRef(3*idxs[j]+l,3*idxs[k]+n) += signs[j] * signs[k] * K(l,n);
                    }
                
                    dFdv.coeffRef(3*idxs[j]+l,3*idxs[k]+l) -= signs[j] * signs[k] * kd;
                }
            }
        }
    }
}

void ImmediateBucklingEnergy::getHessianPattern(const TriMesh& mesh,
                                                vector<SparseTripletd> &triplets) const {

    for (size_t idx=0; idx<pairs_.size(); idx++) {
        int i0 = pairs_[idx].first;
        int i1 = pairs_[idx].second;

        if (i0 == -1 || i1 == -1) {
            continue;
        }

        int idxs[2] = { i0, i1 };
        for (size_t j=0; j<2; ++j)
            for (size_t k=0; k<2; ++k)
                for (size_t l=0; l<3; ++l)
                    for (size_t n=0; n<3; ++n)
                        triplets.push_back(SparseTripletd(3*idxs[j]+l, 3*idxs[k]+n, 1.0));
    }
}

