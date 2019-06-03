// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include <clara.hpp>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>

#include "admm_integrator.hpp"
#include "energy.hpp"
#include "triangle_energies.hpp"
#include "hinge_energy.hpp"
#include "mesh.hpp"


struct CLOptions {
    std::string mesh;
    std::string avatar;

    double density = 0.01;
    double friction = 0.0;
    double damping = 0.0;
    double dt = 1.0 / 30.0;

    int iterations = 10;

    bool showhelp = false;
};


bool step(igl::opengl::glfw::Viewer& viewer,
          ADMM_Integrator& admm,
          SimMesh& mesh,
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
    | clara::Opt(options.avatar, "avatar")
        ["-a"]["--avatar"]
        ("The avatar mesh in OBJ format.")
    | clara::Opt(options.density, "density")
        ["--density"]
        ("The density of the material in g / cm^2. Defaults to 0.01")
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


    SimMesh sim_mesh;

    Eigen::MatrixXd V;
    Eigen::MatrixXd VT;
    Eigen::MatrixXd VN;

    Eigen::MatrixXi FN;

    igl::readOBJ(options.mesh, V, VT, VN, sim_mesh.f, sim_mesh.ft, FN);

    viewer.data().set_mesh(V, sim_mesh.f);
    sim_mesh.idx = viewer.data_list.size() - 1;

    sim_mesh.x = Eigen::VectorXd(3 * V.rows());
    sim_mesh.vt = Eigen::VectorXd(2 * VT.rows());
    sim_mesh.v = Eigen::VectorXd::Zero(3 * V.rows());
    sim_mesh.m = Eigen::VectorXd::Ones(3 * V.rows()) * 0.1;
    for (int i=0; i<V.rows(); i++) {
        sim_mesh.x.segment<3>(3*i) = V.row(i);
    }

    for (int i=0; i<VT.rows(); i++) {
        sim_mesh.vt.segment<2>(2*i) = VT.row(i);
    }

    std::cout << V.rows() << "; " << VT.rows() << std::endl;

    TriMesh avatar;
    if (!options.avatar.empty()) {
        igl::readOBJ(options.avatar, V, avatar.f);
        Eigen::MatrixXd C = Eigen::MatrixXd::Constant(V.rows(), 3, 0.7);
        viewer.append_mesh();
        viewer.data().set_mesh(V, avatar.f);
        viewer.data().set_colors(C);
        avatar.idx = viewer.data_list.size() - 1;

        avatar.x = Eigen::VectorXd(3 * V.rows());
        for (int i=0; i<V.rows(); i++) {
            avatar.x.segment<3>(3*i) = V.row(i);
        }

        avatar.bvh.init(avatar.f, avatar.x, 0.1);
        admm.addAvatar(avatar);
    }


    for (int i=0; i<sim_mesh.f.rows(); i++) {
        std::vector<Eigen::Vector3d> xs;
        for (int j=0; j<3; j++) {
            if (sim_mesh.ft.rows() > 0) {
                Eigen::Vector2d x = sim_mesh.vt.segment<2>(2*sim_mesh.ft(i,j)) * 10.0;
                xs.push_back(Eigen::Vector3d(x[0], x[1], 0.0));
            } else {
                xs.push_back(sim_mesh.x.segment<3>(3*sim_mesh.f(i,j)));
            }
        }
        admm.energies.push_back(std::make_shared<TriangleOrthoStrain>(sim_mesh.f.row(i), xs, 1000000));
    }

    //auto bend_energies = get_edge_energies(sim_mesh);
    //for (auto& e : bend_energies) {
    //    admm.energies.emplace_back(e);
    //}

    viewer.callback_key_down = &key_down;
    viewer.callback_pre_draw = std::bind(&step, std::placeholders::_1, std::ref(admm), std::ref(sim_mesh), options.dt, options.iterations, options.damping, options.friction);

    viewer.launch();
}
