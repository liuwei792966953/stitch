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
    using VecXr = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Real = double;

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

    std::vector<std::array<float, 3>> c; // Colors

    Eigen::VectorXi flags;

    BVH<2> bvh;

    int idx;

    std::unordered_set<int> fixed;

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

    int n_edges() const { return e.rows(); }

    int n_triangles() const { return f.rows(); }

    int n_stitches() const { return s.rows(); }

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

    void reset_colors(const Eigen::Vector3f& def_color) {
        c.resize(x.size() / 3);

        const std::array<float, 3> col = { def_color[0], def_color[1], def_color[2] };

        for (int i=0; i<x.size()/3; i++) {
            c[i] = col;
        }
    }

    void color_flagged(const Eigen::Vector3f& flag_color) {
        const std::array<float, 3> col = { flag_color[0], flag_color[1], flag_color[2] };
        for (int i=0; i<flags.size(); i++) {
            if (flags[i]) {
                c[i] = col;
            }
        }
    }
};

template <typename Real>
struct MeshBase
{
    using Vec2r = Eigen::Matrix<Real, 2, 1>;
    using Vec3r = Eigen::Matrix<Real, 3, 1>;
    using VecXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using Mat3r = Eigen::Matrix<Real, 3, 3>;
    using MatXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
};


class SimMesh : public MeshBase<double>
{
public:
    SimMesh(TriMesh& mesh) : mesh_(mesh) { }

    const TriMesh& tri_mesh() const { return mesh_; }
    TriMesh& tri_mesh() { return mesh_; }

public:
    using Real = double;

    Vec3r ks_stretch = Vec3r(1.0e5, 1.0e5, 1.0e5);
    Vec3r kd_stretch = Vec3r::Zero();
    Vec2r ks_bend = Vec2r(0.01, 0.01);
    Vec2r kd_bend = Vec2r::Zero();

    Real ks_stitch = 1.0e5;
    Real kd_stitch = 1.5e2;

    Real mu = 0.0;


    // Required functions and types
public:
    // Store row index as the iterator
    struct Vertex
    {
        using iterator = int;

        static constexpr int size = 1;
    };

    struct Edge
    {
        using iterator = int;
        
        static constexpr int size = 2;
    };

    struct Triangle
    {
        using iterator = int;

        static constexpr int size = 3;
    };

    struct Stitch 
    {
        using iterator = int;
        
        static constexpr int size = 2;
    };

    struct Diamond
    {
        using iterator = int;
        
        static constexpr int size = 4;
    };

    template <typename Element>
    int n_elements() const { assert(false && "Not implemented!"); return 0; }

    template<>
    int n_elements<Vertex>() const { return mesh_.n_vertices(); }

    template<>
    int n_elements<Edge>() const { return mesh_.n_edges(); }

    template<>
    int n_elements<Triangle>() const { return mesh_.n_triangles(); }

    template<>
    int n_elements<Stitch>() const { return mesh_.n_stitches(); }

    template<>
    int n_elements<Diamond>() const { return mesh_.n_edges(); }

    template <typename Element>
    typename Element::iterator begin() const { return 0; }

    template <typename Element>
    typename Element::iterator end() const { return n_elements<Element>(); }


    int vidx(Vertex::iterator vh) const { return vh; }

    template <typename Element>
    std::array<Vertex::iterator, Element::size> vertices(typename Element::iterator it) const
    { assert(false && "Vertex accessor not implemented!"); }

    template <>
    std::array<Vertex::iterator, Vertex::size> vertices<Vertex>(typename Vertex::iterator it) const
    { return { it }; }

    template <>
    std::array<Vertex::iterator, Edge::size> vertices<Edge>(typename Edge::iterator it) const 
    { return { mesh_.e(it, 0), mesh_.e(it, 1) }; }
    
    template <>
    std::array<Vertex::iterator, Triangle::size> vertices<Triangle>(typename Triangle::iterator it) const 
    { return { mesh_.f(it, 0), mesh_.f(it, 1), mesh_.f(it, 2) }; }

    template <>
    std::array<Vertex::iterator, Stitch::size> vertices<Stitch>(Stitch::iterator it) const 
    { return { mesh_.s(it, 0), mesh_.s(it, 1) }; }

    template <>
    std::array<Vertex::iterator, Diamond::size> vertices<Diamond>(Diamond::iterator it) const  {
        return { mesh_.e (it, 0), mesh_.e (it, 1),
                 mesh_.ei(it, 0), mesh_.ei(it, 1) };
    }

    
    // Vertex
    Vec3r x(Vertex::iterator vh) const // Current vertex position
    { return mesh_.x.segment<3>(3*vh); }

    template <typename Derived>
    void set_x(Vertex::iterator vh, const Eigen::MatrixBase<Derived>& x_new)
    { mesh_.x.segment<3>(3*vh) = x_new; }
    
    Vec3r u(Vertex::iterator vh) const // Undeformed vertex position
    { return Vec3r(mesh_.u[2*vh], mesh_.u[2*vh+1], 0.0); }
    
    Vec3r v(Vertex::iterator vh) const // Current vertex velocity
    { return mesh_.v.segment<3>(3*vh); }
    
    template <typename Derived>
    void set_v(Vertex::iterator vh, const Eigen::MatrixBase<Derived>& v_new)
    { mesh_.v.segment<3>(3*vh) = v_new; }
    
    Real mass(Vertex::iterator vh) const  // Vertex mass
    { return mesh_.m[3*vh]; }
    
    Real friction(Vertex::iterator vh) const { return mu; }

    bool is_fixed(Vertex::iterator vh) const { return mesh_.fixed.count(vh); }

    Real ks_x(Vertex::iterator vh) const { return ks_stretch.x(); }
    Real ks_y(Vertex::iterator vh) const { return ks_stretch.y(); }
    Real ks_shear(Vertex::iterator vh) const { return ks_stretch.z(); }

    Real kd_x(Vertex::iterator vh) const { return kd_stretch.x(); }
    Real kd_y(Vertex::iterator vh) const { return kd_stretch.y(); }
    Real kd_shear(Vertex::iterator vh) const { return kd_stretch.z(); }

    // Edge
    std::array<Vertex::iterator, 2> edge_vertices(Edge::iterator eh) const
    { return { mesh_.e(eh, 0), mesh_.e(eh, 1) }; }

    Real shrinkage(Edge::iterator eh) const
    { return 1.0; }

    Real stretch_ks(Edge::iterator eh) const
    { return ks_stretch.x(); }

    Real stretch_kd(Edge::iterator eh) const
    { return kd_stretch.x(); }

    Real kb_x(Edge::iterator eh) const { return ks_bend.x(); }
    Real kb_y(Edge::iterator eh) const { return ks_bend.y(); }
    Real kbd_x(Edge::iterator eh) const { return kd_bend.x(); }
    Real kbd_y(Edge::iterator eh) const { return kd_bend.y(); }

    // Stitch
    std::array<Vertex::iterator, 2> stitch_vertices(Stitch::iterator sh) const
    { return { mesh_.s(sh, 0), mesh_.s(sh, 1) }; }

    Real stitch_ks(Stitch::iterator sh) const { return ks_stitch; }
    Real stitch_kd(Stitch::iterator sh) const { return kd_stitch; }

    
    std::array<Vertex::iterator, 4> diamond_vertices(Diamond::iterator dh) const {
        return { mesh_.e (dh, 0), mesh_.e (dh, 1),
                 mesh_.ei(dh, 0), mesh_.ei(dh, 1) };
    }

    Real bend_ks(Diamond::iterator dh) const
    { return ks_bend.x(); }

    Real bend_kd(Diamond::iterator dh) const
    { return kd_bend.x(); }

    //Real stretch_ks(Triangle::iterator eh) const
    //{ return ks_stretch.x(); }

    //Real stretch_kd(Triangle::iterator eh) const
    //{ return kd_stretch.x(); }

protected:
    TriMesh& mesh_;
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

    mesh.flags = Eigen::VectorXi::Zero(mesh.x.size() / 3);

    mesh.update_normals();
    mesh.reset_colors(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
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
