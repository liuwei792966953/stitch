// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <array>
#include <unordered_map>

#include "energy.hpp"
#include "mesh.hpp"


// Helper function to create bending energies
std::vector<std::shared_ptr<Energy>> get_edge_energies(TriMesh& mesh, double kb, bool across_stitches=true);

// Ko & Choi bending model
//
class IBM_Energy : public Energy {
public:
    int dim() const { return 3; }

    IBM_Energy(const std::vector<Eigen::Vector3d>& x,
                int idx1, int idx2, double kb)
        : idx1_(idx1), idx2_(idx2) {
        weights_ = Eigen::Vector3d::Constant(kb);

        rest_length_ = (x[3] - x[2]).norm();
    }
    
    void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const {
        for (int i=0; i<3; i++) {
            triplets.push_back(Eigen::Triplet<double>(i, 3 * idx2_ + i,  1.0));
            triplets.push_back(Eigen::Triplet<double>(i, 3 * idx1_ + i, -1.0));
        }
    }

    void project(Eigen::VectorXd &zi) const {
        const double l = zi.norm();
        if (l < rest_length_) {
            zi = 0.5 * (rest_length_ * zi / l + zi);
        }
    }

    Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
        return x.segment<3>(3*idx2_) - x.segment<3>(3*idx1_);
    }

protected:
    int idx1_;
    int idx2_;

    double rest_length_;
};


// A hinge-based bending energy
//
class HingeEnergy : public Energy {
protected:
	typedef Eigen::Matrix<double,9,1> Vector9d;

	int idx1_; // Edge v1
	int idx2_; // Edge v2
	int idx3_; // Opposing v1
	int idx4_; // Opposing v2

        Eigen::Vector4d alpha_;

public:
	int dim() const { return 9; }

	HingeEnergy(const std::vector<Eigen::Vector3d>& x,
	            int idx1, int idx2, int idx3, int idx4)
	    : idx1_(idx1), idx2_(idx2), idx3_(idx3), idx4_(idx4) {

	    weights_ = Eigen::VectorXd::Constant(9, 0.1);

	    Eigen::Vector3d xA = x[2] - x[0];
	    Eigen::Vector3d xB = x[3] - x[0];
	    Eigen::Vector3d xC = Eigen::Vector3d::Zero();
	    Eigen::Vector3d xD = x[1] - x[0];

            double area1 = 0.5 * (xA.cross(xD)).norm();
            double area2 = 0.5 * (xD.cross(xB)).norm();
            double hA = 2.0 * area1 / xD.norm();
            double hB = 2.0 * area2 / xD.norm();

            Eigen::Vector3d nA = (xA - xC).cross(xA - xD);
            Eigen::Vector3d nB = (xB - xD).cross(xB - xC);
            Eigen::Vector3d nC = (xC - xB).cross(xC - xA);
            Eigen::Vector3d nD = (xD - xA).cross(xD - xB);

            alpha_[2] = hB / (hA + hB);
            alpha_[3] = hA / (hA + hB);
            alpha_[0] = -nD.norm() / (nC.norm() + nD.norm());
            alpha_[1] = -nC.norm() / (nC.norm() + nD.norm());
        }

        void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const {
            for (int i=0; i<3; i++) {
                triplets.push_back(Eigen::Triplet<double>(i, 3*idx2_+i,  1.0));
                triplets.push_back(Eigen::Triplet<double>(i, 3*idx1_+i, -1.0));
                
                triplets.push_back(Eigen::Triplet<double>(i+3, 3*idx3_+i,  1.0));
                triplets.push_back(Eigen::Triplet<double>(i+3, 3*idx1_+i, -1.0));
                
                triplets.push_back(Eigen::Triplet<double>(i+6, 3*idx4_+i,  1.0));
                triplets.push_back(Eigen::Triplet<double>(i+6, 3*idx1_+i, -1.0));
            }
        }

        void project(Eigen::VectorXd &zi) const {
            Eigen::Vector3d lam = (alpha_[1] * zi.head<3>() +
                                   alpha_[2] * zi.segment<3>(3) +
                                   alpha_[3] * zi.tail<3>()) /
                (alpha_[1] * alpha_[1] + alpha_[2] * alpha_[2] + alpha_[3] * alpha_[3]);

            Vector9d p;
            p.head<3>()     = zi.head<3>()     - alpha_[1] * lam;
            p.segment<3>(3) = zi.segment<3>(3) - alpha_[2] * lam;
            p.tail<3>()     = zi.tail<3>()     - alpha_[3] * lam;

            zi = 0.5 * (p + zi);
	}

	Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
	    Vector9d dx;
	    dx.segment<3>(0) = x.segment<3>(3*idx2_) - x.segment<3>(3*idx1_);
	    dx.segment<3>(3) = x.segment<3>(3*idx3_) - x.segment<3>(3*idx1_);
	    dx.segment<3>(6) = x.segment<3>(3*idx4_) - x.segment<3>(3*idx1_);
	    return dx;
        }
};


