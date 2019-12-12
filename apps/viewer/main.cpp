// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#include <clara.hpp>

#include "admm_integrator.hpp"
#include "backward_euler_integrator.hpp"
#include "energy.hpp"
#include "hinge_energy.hpp"
#include "immediate_buckling_energy.hpp"
#include "linear_elastic_energy.hpp"
#include "linear_spring_energy.hpp"
#include "mesh.hpp"
#include "nonlinear_elastic_energy.hpp"
#include "stitch_energy.hpp"
#include "triangle_energies.hpp"
#include "xpbd_integrator.hpp"

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
    double ks_x = 1.0e6;
    double ks_y = 1.0e6;
    double kb = 1.0;
    double ks_stitch = 1.0e5;
    double kd_stitch = 1.5e2;
    double damping = 0.0;
    double dt = 1.0 / 30.0;

    int iterations = 10;
    int integrator = 2; // 0 = ADMM, 1 = Backward Euler, 2 = XPBD

    bool showhelp = false;
};


template <typename Integrator>
void update(SimMesh& mesh,
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
        avatar.next_frame(dt);

        struct XPBD_Options {
            double dt_;
            double dt() const { return dt_; }
        };
        XPBD_Options opts{ dt };

        Integrator::template step<SimMesh,AnimatedMesh,XPBD_Options,LinearElasticity<SimMesh>,ImmediateBucklingModel<SimMesh>,StitchSpring<SimMesh>>(mesh, avatar, opts);
        //Integrator::template step<SimMesh,AnimatedMesh,XPBD_Options,LinearSpring<SimMesh>,ImmediateBucklingModel<SimMesh>,StitchSpring<SimMesh>>(mesh, avatar, opts);

        Eigen::MatrixXd V(mesh.tri_mesh().x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = mesh.tri_mesh().x.segment<3>(3*i);
        }

        Eigen::MatrixXd Vs(2*mesh.tri_mesh().s.rows(), 3);
        for (int i=0; i<mesh.tri_mesh().s.rows(); i++) {
            for (int j=0; j<2; j++) {
                Vs.row(2*i+j) = V.row(mesh.tri_mesh().s(i,j));
            }
        }

        polyscope::getSurfaceMesh("cloth")->updateVertexPositions(V);

        mesh.tri_mesh().color_flagged(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
        polyscope::getSurfaceMesh("cloth")->addVertexColorQuantity("color", mesh.tri_mesh().c);

        //if (mesh.s.rows() != 0) {
        //    polyscope::getCurveNetwork("stitches")->updateNodePositions(Vs);
        //}
    }

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


    TriMesh cloth_mesh;
    load_tri_mesh(options.mesh, cloth_mesh, true);

    //if (cloth_mesh.has_uvs()) { cloth_mesh.vt *= 10.0; }

    //for (int i=0; i<cloth_mesh.x.size()/3; i++) {
    //    cloth_mesh.x[3*i+1] += 0.1 * double(rand() / double(RAND_MAX)) - 0.05;
    //}

    Eigen::MatrixXd V(cloth_mesh.x.size() / 3, 3);
    for (int i=0; i<V.rows(); i++) {
        V.row(i) = cloth_mesh.x.segment<3>(3*i);
    }

    cloth_mesh.idx = 0;
    
    cloth_mesh.bvh.init(cloth_mesh.f, cloth_mesh.x);

    cloth_mesh.v = Eigen::VectorXd::Zero(cloth_mesh.x.rows());
    cloth_mesh.m = Eigen::VectorXd::Zero(cloth_mesh.x.rows());

    cloth_mesh.vl = Eigen::VectorXi::Ones(cloth_mesh.x.rows() / 3);
    cloth_mesh.fl = Eigen::VectorXi::Ones(cloth_mesh.f.rows());

    cloth_mesh.u = Eigen::VectorXd(2 * cloth_mesh.x.size() / 3);
    if (cloth_mesh.has_uvs() && cloth_mesh.vt.size()) {
        for (int i=0; i<cloth_mesh.x.size() / 3; i++) {
            int idx = cloth_mesh.uv_index(i);
            if (idx != -1) {
                cloth_mesh.u.segment<2>(2*i) = cloth_mesh.vt.segment<2>(2*idx);
            } else {
                std::cout << "Warning: Cannot find UV for index " << i << std::endl;
                cloth_mesh.u.segment<2>(2*i) = Eigen::Vector2d::Zero();
            }
        }
    } else {
        std::cout << "No UVs..." << std::endl;

        // TODO: Do this in a general way
        size_t zero_index = 1;
        for (int i=0; i<cloth_mesh.x.size() / 3; i++) {
            double* curr = cloth_mesh.u.data() + 2 * i;
            for (int j=0; j<3; j++) {
                if (zero_index != j) {
                    *curr = cloth_mesh.x[3*i+j];
                    curr++;
                }
            }
        }
    }

    /*
    std::unordered_set<int> visited;
    std::vector<std::vector<int>> pieces;

    while (visited.size() != cloth_mesh.x.size() / 3) {
        int start = 0;
        while (visited.count(start)) { start++; }

        std::cout << "Starting piece search at " << start << std::endl;
        std::stack<int> s;
        s.push(start);

        std::vector<int> piece;

        while (!s.empty()) {
            int v = s.top(); s.pop();

            if (!visited.count(v)) {
                for (int i=0; i<cloth_mesh.e.rows(); i++) {
                    for (int j=0; j<2; j++) {
                        if (cloth_mesh.e(i,j) == v && !visited.count(cloth_mesh.e(i,(j+1)%2))) {
                            s.push(cloth_mesh.e(i,(j+1)%2));
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

    for (int i=0; i<cloth_mesh.f.rows(); i++) {
        Vec2d u = (cloth_mesh.u.segment<2>(2*cloth_mesh.f(i,1)) -
                   cloth_mesh.u.segment<2>(2*cloth_mesh.f(i,0)));
        Vec2d v = (cloth_mesh.u.segment<2>(2*cloth_mesh.f(i,2)) -
                   cloth_mesh.u.segment<2>(2*cloth_mesh.f(i,0)));
        const double A = 0.5 * std::fabs(u[0] * v[1] - u[1] * v[0]);

        for (int j=0; j<3; j++) {
            cloth_mesh.m.segment<3>(3*cloth_mesh.f(i,j)) += Eigen::Vector3d::Ones() * A * options.density / 3.0;
        }
    }

    cloth_mesh.fixed.insert(40);
    cloth_mesh.fixed.insert(1680);
    
    cloth_mesh.m.segment<3>(3*40) = Eigen::Vector3d::Constant(1.0e10);
    cloth_mesh.m.segment<3>(3*1680) = Eigen::Vector3d::Constant(1.0e10);

    SimMesh sim_mesh(cloth_mesh);


    AnimatedMesh avatar;
    if (!options.avatar.empty()) {
        load_tri_mesh(options.avatar, avatar);
        avatar.v = Eigen::VectorXd::Zero(avatar.x.size());

        Eigen::MatrixXd V(avatar.x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = avatar.x.segment<3>(3*i);
        }

        avatar.idx = 1;

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
                if (idx1 < cloth_mesh.x.size() / 3 &&
                    idx2 < cloth_mesh.x.size() / 3) {
                    s.push_back(idx1);
                    s.push_back(idx2);
                }
            }

            cloth_mesh.s = Eigen::MatrixXi(s.size() / 2, 2);
            for (int i=0; i<cloth_mesh.s.rows(); i++) {
                for (int j=0; j<2; j++) {
                    cloth_mesh.s(i,j) = s[2*i+j];
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
                    cloth_mesh.vl[i] = l;
                }
            }
        }

        // For now, just assign the face layer as the highest layer of its vertices
        for (int i=0; i<cloth_mesh.f.rows(); i++) {
            cloth_mesh.fl[i] = std::max(std::max(cloth_mesh.vl[cloth_mesh.f(i,0)],
                                               cloth_mesh.vl[cloth_mesh.f(i,1)]),
                                               cloth_mesh.vl[cloth_mesh.f(i,2)]);
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
    polyscope::registerSurfaceMesh("cloth", V, cloth_mesh.f);
    polyscope::getSurfaceMesh("cloth")->setShadeStyle(ShadeStyle::SMOOTH);

    if (avatar.v.size()) {
        Eigen::MatrixXd V(avatar.x.size() / 3, 3);
        for (int i=0; i<V.rows(); i++) {
            V.row(i) = avatar.x.segment<3>(3*i);
        }

        polyscope::registerSurfaceMesh("avatar", V, avatar.f);
        polyscope::getSurfaceMesh("avatar")->setShadeStyle(ShadeStyle::SMOOTH);
    }

    /*
    if (cloth_mesh.s.rows()) {
        Eigen::MatrixXd Vs(2*cloth_mesh.s.rows(), 3);
        std::vector<std::array<size_t, 2>> Ss;
        for (int i=0; i<cloth_mesh.s.rows(); i++) {
            for (int j=0; j<2; j++) {
                Vs.row(2*i+j) = V.row(cloth_mesh.s(i,j));
            }
            Ss.push_back({ size_t(2*i), size_t(2*i+1) });
        }
        polyscope::registerCurveNetwork("stitches", Vs, Ss);
    }
    */

        auto callback = [&]() {
                                    update<XPBD_Integrator>(
                                         sim_mesh,
                                         avatar,
                                         options.dt,
                                         options.iterations,
                                         options.damping,
                                         options.friction);
                                };

        polyscope::state::userCallback = callback;
        polyscope::show();

    return 0;
}
