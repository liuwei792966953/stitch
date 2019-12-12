// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <array>
#include <unordered_map>

#include "energy.hpp"
#include "mesh.hpp"


// Helper function to create bending energies
std::vector<std::shared_ptr<Energy>> get_edge_energies(TriMesh& mesh, double kb, const std::vector<std::tuple<int, int, double>>& angles, bool across_stitches=true);

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
	            int idx1, int idx2, int idx3, int idx4, double kb)
	    : idx1_(idx1), idx2_(idx2), idx3_(idx3), idx4_(idx4) {

	    weights_ = Eigen::VectorXd::Constant(9, kb);

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



class QuadraticEnergy : public Energy {
public:
    int dim() const { return 3; }

    QuadraticEnergy(const std::vector<Eigen::Vector3d>& x,
                int idx1, int idx2, int idx3, int idx4, double kb, double angle=0.0)
            : idx1_(idx1), idx2_(idx2), idx3_(idx3), idx4_(idx4) {
        weights_ = Eigen::VectorXd::Constant(3, kb);

        Eigen::Vector3d x2 = x[2];
        if (std::fabs(angle) > 1.0e-8) {
            Eigen::AngleAxisd aa(angle * M_PI / 180.0, (x[0] - x[1]).normalized());

            x2 = x[0] + aa * (x[2] - x[0]);
            for (int i=0; i<4; i++) {
                if (i == 2)
                    std::cout << "\t" << x2.transpose() << "(" << x[i].transpose() << ")" << std::endl;
                else
                    std::cout << "\t" << x[i].transpose() << std::endl;
            }
        }

        double l01 = (x[0] - x[1]).norm();
        double l02 = (x[0] - x2).norm();
        double l12 = (x[1] - x2).norm();
        double r0 = 0.5 * (l01 + l02 + l12);
        double A0 = std::sqrt(r0 * (r0 - l01) * (r0 - l02) * (r0 - l12));

        double l03 = (x[0] - x[3]).norm();
        double l13 = (x[1] - x[3]).norm();
        double r1 = 0.5 * (l01 + l03 + l13);
        double A1 = std::sqrt(r1 * (r1 - l01) * (r1 - l03) * (r1 - l13));
        weights_ *= std::sqrt(3.0 / (A0 + A1));

        double cot02 = ((l01 * l01) - (l02 * l02) + (l12 * l12)) / (4.0 * A0);
        double cot12 = ((l01 * l01) + (l02 * l02) - (l12 * l12)) / (4.0 * A0);
        double cot03 = ((l01 * l01) - (l03 * l03) + (l13 * l13)) / (4.0 * A1);
        double cot13 = ((l01 * l01) + (l03 * l03) - (l13 * l13)) / (4.0 * A1);

        w_[0] =   cot02 + cot03;
        w_[1] =   cot12 + cot13;
        w_[2] = -(cot02 + cot12);
        w_[3] = -(cot03 + cot13);

        Eigen::Vector3d wx = Eigen::Vector3d::Zero();
        for (size_t i=0; i<4; i++) {
            if (i == 2) wx += x2 * w_[i];
            else wx += x[i] * w_[i];
        }

        n_ = wx.norm();
    }

    void get_reduction(std::vector<Eigen::Triplet<double>> &triplets) const {
        for (int i = 0; i < 3; ++i) {
            triplets.push_back(Eigen::Triplet<double>(i, 3*idx1_+i, w_[0]));
            triplets.push_back(Eigen::Triplet<double>(i, 3*idx2_+i, w_[1]));
            triplets.push_back(Eigen::Triplet<double>(i, 3*idx3_+i, w_[2]));
            triplets.push_back(Eigen::Triplet<double>(i, 3*idx4_+i, w_[3]));
        }
    }

    void project(Eigen::VectorXd &zi) const {
        double l = zi.head<3>().norm();
        if (l > 1.0e-6) {
            zi.head<3>() *= n_ / l;
        }
    }

    Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
        return w_[0] * x.segment<3>(3*idx1_) +
               w_[1] * x.segment<3>(3*idx2_) +
               w_[2] * x.segment<3>(3*idx3_) +
               w_[3] * x.segment<3>(3*idx4_);
    }

protected:
    int idx1_; // Edge v1
    int idx2_; // Edge v2
    int idx3_; // Opposing v1
    int idx4_; // Opposing v2

    Eigen::Vector4d w_;

    double n_;
};



class BendEnergy : public Energy {
public:
    typedef Eigen::Matrix<double,9,1> Vector9d;

    int dim() const { return 9; }

    BendEnergy(const std::vector<Eigen::Vector3d>& x,
                int idx1, int idx2, int idx3, int idx4, double kb, double angle=0.0)
            : idx1_(idx1), idx2_(idx2), idx3_(idx3), idx4_(idx4), angle_(angle*M_PI/180.0) {
        weights_ = Eigen::VectorXd::Constant(9, kb);
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
        Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d p2 = zi.head<3>();
        Eigen::Vector3d p3 = zi.segment<3>(3);
        Eigen::Vector3d p4 = zi.tail<3>();

        Eigen::Vector3d n1 = p2.cross(p3);
        Eigen::Vector3d n2 = p2.cross(p4);

        const double p2p3 = n1.norm();
        const double p2p4 = n2.norm();

        n1 /= p2p3;
        n2 /= p2p4;

        const double d = n1.dot(n2);
        const double num = std::sqrt(1.0 - d*d) * (std::acos(d) - angle_);
        if (std::fabs(num) > 1.0e-8) {
            std::cout << p2p3 << "; " << p2p4 << "; " << d << std::endl;
            std::cout << p2.transpose() << "; " << p3.transpose() << "; " << p4.transpose() << std::endl;

            Eigen::Vector3d q3 = (p2.cross(n2) + n1.cross(p2) * d) / p2p3;
            Eigen::Vector3d q4 = (p2.cross(n1) + n2.cross(p2) * d) / p2p4;
            Eigen::Vector3d q2 = (p3.cross(n2) + n1.cross(p3) * d) / p2p3 -
                                 (p4.cross(n1) + n2.cross(p4) * d) / p2p4;
            Eigen::Vector3d q1 = -q2 - q3 - q4;

            const double factor = num / (weights_[0] * q1.squaredNorm() +
                                         weights_[0] * q2.squaredNorm() +
                                         weights_[0] * q3.squaredNorm() +
                                         weights_[0] * q4.squaredNorm());

            std::cout << "Num: " << num << "; " << d << "; " << factor << std::endl;

            Eigen::Vector3d dp1 = q1 * factor; 
            Eigen::Vector3d dp2 = q2 * factor; 
            Eigen::Vector3d dp3 = q3 * factor; 
            Eigen::Vector3d dp4 = q4 * factor;

            zi.head<3>()     = 0.5 * (zi.head<3>()     + dp2);
            zi.segment<3>(3) = 0.5 * (zi.segment<3>(3) + dp3);
            zi.tail<3>()     = 0.5 * (zi.tail<3>()     + dp4);
        }
    }

    Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
	Vector9d dx;
	dx.segment<3>(0) = x.segment<3>(3*idx2_) - x.segment<3>(3*idx1_);
	dx.segment<3>(3) = x.segment<3>(3*idx3_) - x.segment<3>(3*idx1_);
	dx.segment<3>(6) = x.segment<3>(3*idx4_) - x.segment<3>(3*idx1_);
	return dx;
    }

protected:
    int idx1_; // Edge v1
    int idx2_; // Edge v2
    int idx3_; // Opposing v1
    int idx4_; // Opposing v2

    double angle_;
};


