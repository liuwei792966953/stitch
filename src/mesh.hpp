// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include "bvh.hpp"
#include <Eigen/Core>
#include <igl/edge_flaps.h>
#include <igl/readOBJ.h>
#include <queue>

#include "timer.hpp"

struct TriMesh {
    Eigen::MatrixXi f;
    Eigen::MatrixXi ft;
    Eigen::VectorXd fn;
    Eigen::VectorXd x;
    Eigen::VectorXd u;
    Eigen::VectorXd vt;
    Eigen::VectorXd vn;

    Eigen::MatrixXi e;  // edges (#E x 2)
    Eigen::MatrixXi ef; // edge-face (#E x 2)
    Eigen::MatrixXi ei; // Edge opp vertices (#e x 2)

    Eigen::MatrixXi s; // stitches (#S x 2)

    Eigen::VectorXd v;
    Eigen::VectorXd m;
    Eigen::VectorXi vl;
    Eigen::VectorXi fl;

    BVH<2> bvh;

    int idx;

    bool has_uvs() const {
        return f.rows() == ft.rows() && (x.size() / 3) == (vt.size() / 2);
    }

    // Takes a vertex index and returns an index into vt for that vertex
    int uv_index(int idx) const {
        for (int i=0; i<f.rows(); i++) {
            for (int j=0; j<3; j++) {
                if (f(i,j) == idx) {
                    return ft(i,j);
                }
            }
        }
        return -1;
    }

    int edge_index(int v0, int v1) const {
        for (int i=0; i<e.rows(); i++) {
            if ((e(i,0) == v0 && e(i,1) == v1) ||
                (e(i,0) == v1 && e(i,1) == v0)) {
                return i;
            }
        }
        return -1;
    }

    int n_vertices() const { return x.size() / 3; }

    void update_normals() {
        vn.setZero();

        for (int i=0; i<f.rows(); i++) {
            fn.segment<3>(3*i) = (x.segment<3>(3*f(i,1)) -
                                  x.segment<3>(3*f(i,0))).cross(
                                  x.segment<3>(3*f(i,2)) -
                                  x.segment<3>(3*f(i,0))).normalized();

            for (int j=0; j<3; j++) {
                vn.segment<3>(3*f(i,j)) += fn.segment<3>(3*i);
            }
        }

        for (int i=0; i<n_vertices(); i++) {
            vn.segment<3>(3*i).normalize();
        }
    }
};

inline
void load_tri_mesh(const std::string& fn, TriMesh& mesh, bool get_edges=false) {
    Eigen::MatrixXd V;
    Eigen::MatrixXd VT;
    Eigen::MatrixXd VN;
    Eigen::MatrixXi FN;

    igl::readOBJ(fn, V, VT, VN, mesh.f, mesh.ft, FN);

    mesh.x = Eigen::VectorXd(3 * V.rows());
    mesh.vt = Eigen::VectorXd(2 * VT.rows());

    for (int i=0; i<V.rows(); i++) {
        mesh.x.segment<3>(3*i) = V.row(i);
    }

    for (int i=0; i<VT.rows(); i++) {
        mesh.vt.segment<2>(2*i) = VT.row(i);
    }

    if (get_edges) {
        Eigen::VectorXi emap;
        igl::edge_flaps(mesh.f, mesh.e, emap, mesh.ef, mesh.ei);
    }

    mesh.fn = Eigen::VectorXd(mesh.f.rows() * 3);
    mesh.vn = Eigen::VectorXd(mesh.x.size());

    mesh.update_normals();
}

struct AnimatedMesh : TriMesh {

    void next_frame(double dt) {
        if (vertex_data.size()) {
            Eigen::VectorXd old_x = x;

            x = Eigen::Map<Eigen::VectorXd>(vertex_data.front().data(), x.size());
            vertex_data.pop();

            bvh.refit(f, x, 2.5);

            v = (x - old_x) / dt;
        } else {
            v.setZero();
        }
    }
    
    std::queue<std::vector<double>> vertex_data;
};
