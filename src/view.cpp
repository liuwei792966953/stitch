// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include <clara.hpp>
#include <igl/opengl/glfw/Viewer.h>

#include "admm_integrator.hpp"
#include "backward_euler_integrator.hpp"
#include "energy.hpp"
#include "hinge_energy.hpp"
#include "immediate_buckling_energy.hpp"
#include "nonlinear_elastic_energy.hpp"
#include "stitch_energy.hpp"
#include "triangle_energies.hpp"
#include "mesh.hpp"

#include "polyscope/curve_network.h"
#include "polyscope/surface_mesh.h"

bool is_running = false;



struct CLOptions {
    std::string mesh;
    std::string avatar;
    std::string avatar_animation;
    std::string stitches;
    std::string bend_angles;
    std::string layers;

    double density = 0.02;
    double friction = 0.0;
    double ks_x = 1.0e4;
    double ks_y = 1.0e4;
    double kb = 1.0;
    double ks_stitch = 1.0e4;
    double kd_stitch = 1.5e2;
    double damping = 0.0;
    double dt = 1.0 / 30.0;

    int iterations = 10;
    int integrator = 0; // 0 = ADMM, 1 = Backward Euler

    bool showhelp = false;
};


template <typename Integrator>
bool step(igl::opengl::glfw::Viewer& viewer,
          Integrator& integrator,
          TriMesh& mesh,
          AnimatedMesh& avatar,
          double dt,
          int iterations,
          double kd,
          double mu)
{
    if (viewer.core.is_animating) {
        integrator.step(mesh, dt, iterations, kd, mu);

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

template <typename Integrator>
void update(Integrator& integrator,
        TriMesh& mesh,
          AnimatedMesh& avatar,
          double dt,
          int iterations,
          double kd,
          double mu)
{
    const std::string label = std::string(is_running ? "Stop" : "Start") +
                                std::string(" simulation");
    if (ImGui::Button(label.c_str())) {
        is_running = !is_running;
    }

    if (is_running) {
        integrator.step(mesh, dt, iterations, kd, mu);

        Eigen::MatrixXd V(mesh.x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = mesh.x.segment<3>(3*i);
        }

        Eigen::MatrixXd Vs(2*mesh.s.rows(), 3);
        for (int i=0; i<mesh.s.rows(); i++) {
            for (int j=0; j<2; j++) {
                Vs.row(2*i+j) = V.row(mesh.s(i,j));
            }
        }

        polyscope::getSurfaceMesh("cloth")->updateVertexPositions(V);
        polyscope::getCurveNetwork("stitches")->updateNodePositions(Vs);
    }

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
    | clara::Opt(options.avatar_animation, "avatar-animation")
        ["-aa"]["--avatar-animation"]
        ("Binary file containing avatar animation data.")
    | clara::Opt(options.bend_angles, "bend_angles")
        ["-ba"]["--bend-angles"]
        ("A file that specifies bend angles, where each line is v0 v1 angle.")
    | clara::Opt(options.layers, "layers")
        ["-l"]["--layers"]
        ("A file that specifies vertex layers, where each line is v_start v_end l.")
    | clara::Opt(options.density, "density")
        ["--density"]
        ("The density of the material in g / cm^2. Defaults to 0.01")
    | clara::Opt(options.ks_x, "ks_x")
        ["--ksx"]
        ("The stretching resistance in the x-direction. Defaults to 1.0e4.")
    | clara::Opt(options.ks_y, "ks_y")
        ["--ksy"]
        ("The stretching resistance in the y-direction. Defaults to 1.0e4.")
    | clara::Opt(options.kb, "kb")
        ["--kb"]
        ("The bending resistance. Defaults to 1.0.")
    | clara::Opt(options.ks_stitch, "ks-stitch")
        ["--ks-stitch"]
        ("The stitch stiffness. Defaults to 1.0e6.")
    | clara::Opt(options.kd_stitch, "kd-stitch")
        ["--kd-stitch"]
        ("The stitch damping. Defaults to 1.5e2.")
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
    | clara::Opt(options.integrator, "integrator")
        ["--integrator"]
        ("Integrator. 0 -> ADMM, 1 -> Backward Euler.")
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



    TriMesh sim_mesh;
    load_tri_mesh(options.mesh, sim_mesh, true);

    //if (sim_mesh.has_uvs()) { sim_mesh.vt *= 10.0; }

    //for (int i=0; i<sim_mesh.x.size()/3; i++) {
    //    sim_mesh.x[3*i+1] += 0.1 * double(rand() / double(RAND_MAX)) - 0.05;
    //}

    Eigen::MatrixXd V(sim_mesh.x.size() / 3, 3);
    for (int i=0; i<V.rows(); i++) {
        V.row(i) = sim_mesh.x.segment<3>(3*i);
    }

    viewer.data().set_mesh(V, sim_mesh.f);
    sim_mesh.idx = viewer.data_list.size() - 1;
    
    sim_mesh.bvh.init(sim_mesh.f, sim_mesh.x);

    sim_mesh.v = Eigen::VectorXd::Zero(sim_mesh.x.rows());
    sim_mesh.m = Eigen::VectorXd::Zero(sim_mesh.x.rows());

    sim_mesh.vl = Eigen::VectorXi::Ones(sim_mesh.x.rows() / 3);
    sim_mesh.fl = Eigen::VectorXi::Ones(sim_mesh.f.rows());

    sim_mesh.u = Eigen::VectorXd(2 * sim_mesh.x.size() / 3);
    if (sim_mesh.has_uvs() && sim_mesh.vt.size()) {
        for (int i=0; i<sim_mesh.x.size() / 3; i++) {
            int idx = sim_mesh.uv_index(i);
            if (idx != -1) {
                sim_mesh.u.segment<2>(2*i) = sim_mesh.vt.segment<2>(2*idx);
            } else {
                std::cout << "Warning: Cannot find UV for index " << i << std::endl;
                sim_mesh.u.segment<2>(2*i) = Eigen::Vector2d::Zero();
            }
        }
    } else {
        std::cout << "No UVs..." << std::endl;

        // TODO: Do this in a general way
        size_t zero_index = 1;
        for (int i=0; i<sim_mesh.x.size() / 3; i++) {
            double* curr = sim_mesh.u.data() + 2 * i;
            for (int j=0; j<3; j++) {
                if (zero_index != j) {
                    *curr = sim_mesh.x[3*i+j];
                    curr++;
                }
            }
        }
    }

    /*
    std::unordered_set<int> visited;
    std::vector<std::vector<int>> pieces;

    while (visited.size() != sim_mesh.x.size() / 3) {
        int start = 0;
        while (visited.count(start)) { start++; }

        std::cout << "Starting piece search at " << start << std::endl;
        std::stack<int> s;
        s.push(start);

        std::vector<int> piece;

        while (!s.empty()) {
            int v = s.top(); s.pop();

            if (!visited.count(v)) {
                for (int i=0; i<sim_mesh.e.rows(); i++) {
                    for (int j=0; j<2; j++) {
                        if (sim_mesh.e(i,j) == v && !visited.count(sim_mesh.e(i,(j+1)%2))) {
                            s.push(sim_mesh.e(i,(j+1)%2));
                        }
                    }
                }

                visited.insert(v);
            }
        }

        pieces.emplace_back(piece);
    }
    std::cout << "Found " << pieces.size() << " pieces." << std::endl;
    */

    for (int i=0; i<sim_mesh.f.rows(); i++) {
        Vec2d u = (sim_mesh.u.segment<2>(2*sim_mesh.f(i,1)) -
                   sim_mesh.u.segment<2>(2*sim_mesh.f(i,0)));
        Vec2d v = (sim_mesh.u.segment<2>(2*sim_mesh.f(i,2)) -
                   sim_mesh.u.segment<2>(2*sim_mesh.f(i,0)));
        const Real A = 0.5 * std::fabs(u[0] * v[1] - u[1] * v[0]);

        for (int j=0; j<3; j++) {
            sim_mesh.m.segment<3>(3*sim_mesh.f(i,j)) += Eigen::Vector3d::Ones() * A * options.density / 3.0;
        }
    }


    AnimatedMesh avatar;
    if (!options.avatar.empty()) {
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

        if (!options.avatar_animation.empty()) {
            std::ifstream in(options.avatar_animation, std::ios::binary);
            if (in) {
                in.seekg(0, std::ios::end);
                const size_t num_elements = in.tellg() / sizeof(float);
                in.seekg(0, std::ios::beg);

                std::vector<float> data(num_elements);
                in.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(float));

                size_t curr_idx = 0;
                while (curr_idx + avatar.v.size() <= num_elements) {
                    avatar.vertex_data.emplace(data.data()+curr_idx,
                                               data.data()+curr_idx+avatar.v.size());
                    curr_idx += avatar.v.size();
                }
                std::cout << "Read " << avatar.vertex_data.size() << " frames." << std::endl;
            }
        }
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

    if (!options.layers.empty()) {
        std::ifstream in(options.layers);
        if (in) {
            std::string line;
            while (std::getline(in, line)) {
                std::stringstream str(line);

                int v0, v1, l;
                str >> v0 >> v1 >> l;
                for (int i=v0; i<=v1; i++) {
                    sim_mesh.vl[i] = l;
                }
            }
        }

        // For now, just assign the face layer as the highest layer of its vertices
        for (int i=0; i<sim_mesh.f.rows(); i++) {
            sim_mesh.fl[i] = std::max(std::max(sim_mesh.vl[sim_mesh.f(i,0)],
                                               sim_mesh.vl[sim_mesh.f(i,1)]),
                                               sim_mesh.vl[sim_mesh.f(i,2)]);
        }
    }

    std::vector<std::tuple<int, int, double>> angles;
    if (!options.bend_angles.empty()) {
        std::ifstream in(options.bend_angles);
        if (in) {
            std::string line;
            while (std::getline(in, line)) {
                std::stringstream str(line);

                int idx1, idx2;
                double angle;
                str >> idx1 >> idx2 >> angle;
                angles.push_back({ idx1, idx2, angle });
            }
        }
    }

    polyscope::options::alwaysRedraw = true;
    polyscope::options::autocenterStructures = false;
    polyscope::view::windowWidth = 1024;
    polyscope::view::windowHeight = 768;

    polyscope::init();
    polyscope::registerSurfaceMesh("cloth", V, sim_mesh.f);
    polyscope::getSurfaceMesh("cloth")->setShadeStyle(ShadeStyle::SMOOTH);

    if (avatar.v.size()) {
        Eigen::MatrixXd V(avatar.x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = avatar.x.segment<3>(3*i);
        }

        polyscope::registerSurfaceMesh("avatar", V, avatar.f);
        polyscope::getSurfaceMesh("avatar")->setShadeStyle(ShadeStyle::SMOOTH);
    }

    if (sim_mesh.s.rows()) {
        Eigen::MatrixXd Vs(2*sim_mesh.s.rows(), 3);
        std::vector<std::array<size_t, 2>> Ss;
        for (int i=0; i<sim_mesh.s.rows(); i++) {
            for (int j=0; j<2; j++) {
                Vs.row(2*i+j) = V.row(sim_mesh.s(i,j));
            }
            Ss.push_back({ size_t(2*i), size_t(2*i+1) });
        }
        polyscope::registerCurveNetwork("stitches", Vs, Ss);
    }


    if (options.integrator == 0) {
        ADMM_Integrator admm;

        if (avatar.x.size()) {
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

        auto bend_energies = get_edge_energies(sim_mesh, options.kb, angles, true);
        for (auto& e : bend_energies) {
            admm.energies.emplace_back(e);
        }

        for (int i=0; i<sim_mesh.s.rows(); i++) {
            admm.energies.push_back(std::make_shared<StitchEnergy>(sim_mesh.s(i,0),
                                                                   sim_mesh.s(i,1)));
        }

        /*
        auto pre_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
                                    return step(viewer,
                                         admm,
                                         sim_mesh,
                                         avatar,
                                         options.dt,
                                         options.iterations,
                                         options.damping,
                                         options.friction);
                                };

        viewer.callback_key_down = &key_down;
        viewer.callback_pre_draw = pre_draw;
        viewer.launch();
        */
        
        auto callback = [&]() {
                                    update(admm,
                                         sim_mesh,
                                         avatar,
                                         options.dt,
                                         options.iterations,
                                         options.damping,
                                         options.friction);
                                };

        polyscope::state::userCallback = callback;
        polyscope::show();

    } else if (options.integrator == 1) {
        BackwardEulerIntegrator be;
        
        if (avatar.x.size()) {
            be.addAvatar(avatar);
        }

        be.energies.emplace_back(std::make_unique<NonlinearElasticEnergy>(options.ks_x, options.ks_y));
        be.energies.emplace_back(std::make_unique<ImmediateBucklingEnergy>(options.kb));
        be.energies.emplace_back(std::make_unique<StitchSpringEnergy>(options.ks_stitch, options.kd_stitch));

        /*
        auto pre_draw = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
                                    return step(viewer,
                                         be,
                                         sim_mesh,
                                         avatar,
                                         options.dt,
                                         options.iterations,
                                         options.damping,
                                         options.friction);
                                };

        viewer.callback_key_down = &key_down;
        viewer.callback_pre_draw = pre_draw;
        viewer.launch();
        */

        auto callback = [&]() {
                                    update(be,
                                         sim_mesh,
                                         avatar,
                                         options.dt,
                                         options.iterations,
                                         options.damping,
                                         options.friction);
                                };

        polyscope::state::userCallback = callback;
        polyscope::show();
    }

    return 0;
}
