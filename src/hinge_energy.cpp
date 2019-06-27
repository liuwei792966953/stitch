// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include "hinge_energy.hpp"



// Align xs[4] - xs[3] to xs[1] - xs[0] and return in new space
std::vector<Eigen::Vector3d> get_aligned(const std::vector<Eigen::Vector3d>& xs) {
    Eigen::Vector3d e1 = xs[1] - xs[0];
    Eigen::Vector3d e2 = xs[4] - xs[3];

    Eigen::Vector3d e3 = xs[5] - xs[3];

    if ((e1 - e2).squaredNorm() > 1.0e-8) {
        Eigen::Quaterniond q;
        q.FromTwoVectors(e2, e1);

        // Now rotate other vector by the same amount
        e3 = q * e3;
    }

    std::vector<Eigen::Vector3d> xs_new;
    xs_new.push_back(xs[0]);
    xs_new.push_back(xs[1]);
    xs_new.push_back(xs[2]);
    xs_new.push_back(xs[0] + e3);

    return xs_new;
}


std::vector<std::shared_ptr<Energy>> get_edge_energies(TriMesh& mesh, double kb, bool across_stitches) {
    std::vector<std::shared_ptr<Energy>> energies;

    std::array<int,4> idxs;
    for (int i=0; i<mesh.ei.rows(); i++) {
        if (mesh.ei(i,0) != -1 && mesh.ei(i,1) != -1) {
            idxs[0] = mesh.e(i,0);
            idxs[1] = mesh.e(i,1);
            idxs[2] = mesh.ei(i,0);
            idxs[3] = mesh.ei(i,1);

            std::vector<Eigen::Vector3d> xs;
            for (int i=0; i<4; i++) {
                if (mesh.has_uvs()) {
                    auto uv = mesh.vt.segment<2>(2*mesh.uv_index(idxs[i]));
                    xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
                } else {
                    xs.push_back(mesh.x.segment<3>(3*idxs[i]));
                }
            }
            
            energies.push_back(std::make_shared<IBM_Energy>(xs, idxs[2], idxs[3], kb));
            //energies.push_back(std::make_shared<HingeEnergy>(xs, idxs[0], idxs[1], idxs[2], idxs[3]));
        }
    }

    if (across_stitches) {
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

            // 6 positions, e00, e01, o0, e10, e11, o1

            std::vector<Eigen::Vector3d> xs;
            if (mesh.has_uvs()) {
                auto uv = mesh.vt.segment<2>(2*mesh.uv_index(v0));
                xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(v1));
                xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(o0));
                xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(w0));
                xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(w1));
                xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
                uv = mesh.vt.segment<2>(2*mesh.uv_index(o1));
                xs.push_back(Eigen::Vector3d(uv[0], uv[1], 0.0));
            } else {
                xs.push_back(mesh.x.segment<3>(3*v0));
                xs.push_back(mesh.x.segment<3>(3*v1));
                xs.push_back(mesh.x.segment<3>(3*o0));
                xs.push_back(mesh.x.segment<3>(3*w0));
                xs.push_back(mesh.x.segment<3>(3*w1));
                xs.push_back(mesh.x.segment<3>(3*o1));
            }

            xs = get_aligned(xs);

            energies.push_back(std::make_shared<IBM_Energy>(xs, o0, o1, kb));
        }
    }

    std::cout << "Created " << energies.size() << " bend energies." << std::endl;

    return energies;
}


