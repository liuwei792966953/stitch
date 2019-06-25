// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include <clara.hpp>
#include <igl/opengl/glfw/Viewer.h>

#include "admm_integrator.hpp"
#include "energy.hpp"
#include "hinge_energy.hpp"
#include "stitch_energy.hpp"
#include "triangle_energies.hpp"
#include "mesh.hpp"


struct CLOptions {
    std::string mesh;
    std::string avatar;
    std::string stitches;

    double density = 0.02;
    double friction = 0.0;
    double ks_x = 1.0e7;
    double ks_y = 1.0e6;
    double kb = 100.0;
    double damping = 0.0;
    double dt = 1.0 / 30.0;

    int iterations = 10;

    bool showhelp = false;
};


bool step(igl::opengl::glfw::Viewer& viewer,
          ADMM_Integrator& admm,
          TriMesh& mesh,
          AnimatedMesh& avatar,
          double dt,
          int iterations,
          double kd,
          double mu)
{
    if (viewer.core.is_animating) {
        admm.step(mesh, dt, iterations, kd, mu);

        Eigen::MatrixXd V(mesh.x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = mesh.x.segment<3>(3*i);
        }

        viewer.data_list[mesh.idx].set_vertices(V);

        if (avatar.x.size()) {
            V = Eigen::MatrixXd(avatar.x.size() / 3, 3);
            for (int i=0; i<V.rows(); i++) {
                V.row(i) = avatar.x.segment<3>(3*i);
            }
        
            viewer.data_list[avatar.idx].set_vertices(V);
        }
    }

    return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
  switch(key)
  {
    case ' ':
      viewer.core.is_animating = !viewer.core.is_animating;
      break;
  }
  return true;
}

int main(int argc, char *argv[])
{
    CLOptions options;

    auto cli
    = clara::Arg(options.mesh, "mesh")
        ("The mesh to simulate in OBJ format.")
    | clara::Opt(options.stitches, "stitches")
        ["-s"]["--stitches"]
        ("A file defining the vertex-vertex stitches, one per line.")
    | clara::Opt(options.avatar, "avatar")
        ["-a"]["--avatar"]
        ("The avatar mesh in OBJ format.")
    | clara::Opt(options.density, "density")
        ["--density"]
        ("The density of the material in g / cm^2. Defaults to 0.01")
    | clara::Opt(options.ks_x, "ks_x")
        ["--ksx"]
        ("The stretching resistance in the x-direction. Defaults to 1.0e7.")
    | clara::Opt(options.ks_y, "ks_y")
        ["--ksy"]
        ("The stretching resistance in the y-direction. Defaults to 1.0e6.")
    | clara::Opt(options.kb, "kb")
        ["--kb"]
        ("The bending resistance. Defaults to 100.0.")
    | clara::Opt(options.friction, "friction")
        ["-f"]["--friction"]
        ("The friction parameter. Value between (0.0, inf). Defaults to 0.0.")
    | clara::Opt(options.damping, "damping")
        ["-d"]["--damping"]
        ("Global damping parameter in the range (0.0, 1.0). Defaults to 0.0.")
    | clara::Opt(options.dt, "dt")
        ["--dt"]
        ("Timestep for the simulation. Defaults is 1/30.")
    | clara::Opt(options.iterations, "iterations")
        ["-i"]["--iterations"]
        ("Number of internal iterations to run the ADMM integrator. Default is 10.")
    | clara::Help(options.showhelp);

    
    auto result = cli.parse(clara::Args(argc, argv));

    if (!result) {
        std::cerr << "Could not parse CLI: " << result.errorMessage() << std::endl;
        return 0;
    }

    //if (options.mesh.empty()) {
    //    cli.writeToStream(std::cout);
    //    return 1;
    //}

    if (options.showhelp) {
        cli.writeToStream(std::cout);
        return 0;
    }

    igl::opengl::glfw::Viewer viewer;

    ADMM_Integrator admm;


    TriMesh sim_mesh;
    load_tri_mesh(options.mesh, sim_mesh, true);

    Eigen::MatrixXd V(sim_mesh.x.size() / 3, 3);
    for (int i=0; i<V.rows(); i++) {
        V.row(i) = sim_mesh.x.segment<3>(3*i);
    }

    viewer.data().set_mesh(V, sim_mesh.f);
    sim_mesh.idx = viewer.data_list.size() - 1;

    sim_mesh.v = Eigen::VectorXd::Zero(sim_mesh.x.rows());
    sim_mesh.m = Eigen::VectorXd::Zero(sim_mesh.x.rows());

    for (int i=0; i<sim_mesh.f.rows(); i++) {
        double a = (sim_mesh.x.segment<3>(3*sim_mesh.f(i,1)) - sim_mesh.x.segment<3>(3*sim_mesh.f(i,0))).cross(
                    sim_mesh.x.segment<3>(3*sim_mesh.f(i,2)) - sim_mesh.x.segment<3>(3*sim_mesh.f(i,0))).norm() * 0.5;

        for (int j=0; j<3; j++) {
            sim_mesh.x.segment<3>(3*sim_mesh.f(i,j)) += Eigen::Vector3d::Ones() * a * options.density;
        }
    }


    AnimatedMesh avatar;
    if (!options.avatar.empty()) {
        if (options.avatar.find("avatar") != std::string::npos) {
            for (int i=1; i<122; i++) {
                std::string nbr = std::to_string(i);
                nbr.insert(nbr.begin(), 3 - nbr.size(), '0');
                avatar.obj_files.push_back("/Users/dharmon/Desktop/avatar/avatar_" + nbr + ".obj");
            }
        }

        load_tri_mesh(options.avatar, avatar);
        avatar.v = Eigen::VectorXd::Zero(avatar.x.size());

        Eigen::MatrixXd V(avatar.x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = avatar.x.segment<3>(3*i);
        }

        Eigen::MatrixXd C = Eigen::MatrixXd::Constant(V.rows(), 3, 0.7);
        viewer.append_mesh();
        viewer.data().set_mesh(V, avatar.f);
        viewer.data().set_colors(C);
        avatar.idx = viewer.data_list.size() - 1;

        avatar.bvh.init(avatar.f, avatar.x, 2.5);
        admm.addAvatar(avatar);
    }


    for (int i=0; i<sim_mesh.f.rows(); i++) {
        std::vector<Eigen::Vector3d> xs;
        for (int j=0; j<3; j++) {
            if (sim_mesh.has_uvs()) {
                Eigen::Vector2d x = sim_mesh.vt.segment<2>(2*sim_mesh.uv_index(sim_mesh.f(i,j)));
                xs.push_back(Eigen::Vector3d(x[0], x[1], 0.0));
            } else {
                xs.push_back(sim_mesh.x.segment<3>(3*sim_mesh.f(i,j)));
            }
        }
        admm.energies.push_back(std::make_shared<TriangleOrthoStrain>(sim_mesh.f.row(i), xs,
                    options.ks_x, options.ks_y));
    }

    if (!options.stitches.empty()) {
        std::ifstream in(options.stitches);
        if (in) {
            std::vector<int> s;

            std::string line;
            while (std::getline(in, line)) {
                std::stringstream str(line);

                int idx1, idx2;
                str >> idx1 >> idx2;
                if (idx1 < sim_mesh.x.size() / 3 &&
                    idx2 < sim_mesh.x.size() / 3) {
                    s.push_back(idx1);
                    s.push_back(idx2);
                    admm.energies.push_back(std::make_shared<StitchEnergy>(idx1, idx2));
                }
            }

            sim_mesh.s = Eigen::MatrixXi(s.size() / 2, 2);
            for (int i=0; i<sim_mesh.s.rows(); i++) {
                for (int j=0; j<2; j++) {
                    sim_mesh.s(i,j) = s[2*i+j];
                }
            }
        }
    }

    auto bend_energies = get_edge_energies(sim_mesh, options.kb, false);
    for (auto& e : bend_energies) {
        admm.energies.emplace_back(e);
    }

    viewer.callback_key_down = &key_down;
    viewer.callback_pre_draw = std::bind(&step, std::placeholders::_1, std::ref(admm), std::ref(sim_mesh), std::ref(avatar), options.dt, options.iterations, options.damping, options.friction);

    viewer.launch();
}
