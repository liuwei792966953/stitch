#pragma once

#include <unordered_map>

#include "energy.hpp"
#include "mesh.hpp"


// A hinge-based bending energy
//
class HingeEnergy : public Energy {
protected:
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,9,1> Vec9;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;

	int idx1_; // Edge v1
	int idx2_; // Edge v2
	int idx3_; // Opposing v1
	int idx4_; // Opposing v2

	double rest_angle_ = 0.0;
	
	double weight_;

        Eigen::Vector4d alpha_;

public:
	int get_dim() const { return 9; }
	double get_weight() const {
            return weight_;
	}

	// idx1 / idx2 are opposing vertices, idx3 / idx4 is edge
	HingeEnergy(const Eigen::VectorXd& x, int idx1, int idx2, int idx3, int idx4) : idx1_(idx1), idx2_(idx2), idx3_(idx3), idx4_(idx4), weight_(0.1) {

            Eigen::Vector3d xA = x.segment<3>(3*idx1_) - x.segment<3>(3*idx3_);
            Eigen::Vector3d xB = x.segment<3>(3*idx2_) - x.segment<3>(3*idx3_);
            Eigen::Vector3d xC = Eigen::Vector3d::Zero();
            Eigen::Vector3d xD = x.segment<3>(3*idx4_) - x.segment<3>(3*idx3_);

            double area1 = 0.5 * (xA.cross(xD)).norm();
            double area2 = 0.5 * (xD.cross(xB)).norm();
            double hA = 2.0 * area1 / xD.norm();
            double hB = 2.0 * area2 / xD.norm();

            Eigen::Vector3d nA = (xA - xC).cross(xA - xD);
            Eigen::Vector3d nB = (xB - xD).cross(xB - xC);
            Eigen::Vector3d nC = (xC - xB).cross(xC - xA);
            Eigen::Vector3d nD = (xD - xA).cross(xD - xB);

            alpha_[0] = hB / (hA + hB);
            alpha_[1] = hA / (hA + hB);
            alpha_[2] = -nD.norm() / (nC.norm() + nD.norm());
            alpha_[3] = -nC.norm() / (nC.norm() + nD.norm());
        }

        void get_reduction(std::vector< Eigen::Triplet<double> > &triplets){
            int col0 = 3 * idx1_;
            int col1 = 3 * idx2_;
            int col2 = 3 * idx3_;
            int col3 = 3 * idx4_;

            triplets.push_back(Eigen::Triplet<double>(0, col0, 1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(1, col0+1, 1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(2, col0+2, 1.0 ) );

            triplets.push_back(Eigen::Triplet<double>(0, col2, -1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(1, col2+1, -1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(2, col2+2, -1.0 ) );

            triplets.push_back(Eigen::Triplet<double>(3, col3, 1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(4, col3+1, 1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(5, col3+2, 1.0 ) );

            triplets.push_back(Eigen::Triplet<double>(3, col2, -1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(4, col2+1, -1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(5, col2+2, -1.0 ) );

            triplets.push_back(Eigen::Triplet<double>(6, col1, 1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(7, col1+1, 1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(8, col1+2, 1.0 ) );

            triplets.push_back(Eigen::Triplet<double>(6, col2, -1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(7, col2+1, -1.0 ) );
            triplets.push_back(Eigen::Triplet<double>(8, col2+2, -1.0 ) );
        }

        void prox(Eigen::VectorXd &zi){
            Eigen::Vector3d c1 = zi.segment<3>(0);
            Eigen::Vector3d c2 = zi.segment<3>(3);
            Eigen::Vector3d c3 = zi.segment<3>(6);

            Eigen::Vector3d lam = 2.0 * (alpha_[0]*c1 + alpha_[3]*c2 + alpha_[1]*c3) / (alpha_[0]*alpha_[0] + alpha_[3]*alpha_[3] + alpha_[1]*alpha_[1]);

            Vec9 p;
            p.segment<3>(0) = c1 - 0.5 * alpha_[0] * lam;
            p.segment<3>(3) = c2 - 0.5 * alpha_[3] * lam;
            p.segment<3>(6) = c3 - 0.5 * alpha_[1] * lam;

	    zi.head<9>() = ( 1.0 / (weight_ * weight_ + weight_ * weight_)) * (weight_ * weight_ * p + weight_ * weight_ * (zi.head<9>()));
	}

	Eigen::VectorXd reduce(const Eigen::VectorXd& x) const {
	    Vec9 Dx;
	    Dx.segment<3>(0) = x.segment<3>(3*idx1_) - x.segment<3>(3*idx3_);
	    Dx.segment<3>(3) = x.segment<3>(3*idx4_) - x.segment<3>(3*idx3_);
	    Dx.segment<3>(6) = x.segment<3>(3*idx2_) - x.segment<3>(3*idx3_);
	    return Dx;
        }
};

struct sint2 {
		sint2(){}
		sint2( int a, int b ){
			sorted_v[0]=a; sorted_v[1]=b;
			if( b < a ){ sorted_v[0]=b; sorted_v[1]=a; }
			orig_v[0]=a; orig_v[1]=b;
		}
		bool operator==(const sint2 &a) const {
			return (sorted_v[0] == a.sorted_v[0]
				&& sorted_v[1] == a.sorted_v[1]
			);
		}
		int operator[](const int i) const {
			return orig_v[i];
		}
		int sorted_v[2]; // vertices SORTED
		int orig_v[2]; // original vertices
	};

namespace std {

template <> struct hash<sint2> {
		size_t operator()(const sint2& v) const	{
			int a[2] = { v.sorted_v[0], v.sorted_v[1] };
			unsigned char *in = reinterpret_cast<unsigned char*>(a);
			unsigned int ret = 2654435761u;
			for(unsigned int i = 0; i < (2 * sizeof(int)); ++i)
				ret = (ret * 2654435761u) ^ *in++;
			return ret;
		}
	};
}


std::vector<std::shared_ptr<Energy>> get_edge_energies(TriMesh& mesh) {
    std::vector<std::shared_ptr<Energy>> energies;

    std::unordered_map<sint2, std::pair<int,int>> faces;

    for (int face_idx=0; face_idx<mesh.f.rows(); face_idx++) {
        for (int i=0; i<3; i++) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;

            int idx1 = mesh.f(face_idx, i);
            int idx2 = mesh.f(face_idx, j);

            auto h = sint2(idx1, idx2);
            auto p = faces.emplace(h, std::pair<int,int>(-1, -1));
            std::pair<int,int>& opps = p.first->second;
            if (opps.first == -1) {
                opps.first = face_idx;
            } else if (opps.second == -1) {
                opps.second = face_idx;
            }
        }
    }

    int count = 0;
    for (const auto& p : faces) {
        if (p.second.first != -1 && p.second.second != -1) {
            count++;

            const Eigen::RowVector3i& f1 = mesh.f.row(p.second.first);
            const Eigen::RowVector3i& f2 = mesh.f.row(p.second.second);

            Eigen::Vector4i idxs(-1, -1, -1, -1);

            int curr_edge_idx = 0;
            for (int i=0; i<3; i++) {
                if (std::find(f2.data(), f2.data()+3, f1[i]) != (f2.data()+3)) {
                    idxs[2+curr_edge_idx] = f1[i];
                    curr_edge_idx++;
                } else {
                    idxs[0] = f1[i];
                }
            }

            // Find f2 idx not in the list
            for (int i=0; i<3; i++) {
                if (std::find(idxs.data(), idxs.data()+4, f2[i]) == (idxs.data()+4)) {
                    idxs[1] = f2[i];
                    break;
                }
            }

            //std::cout << idxs.transpose() << "; " << f1.transpose() << "; " << f2.transpose() << std::endl;
            energies.push_back(std::make_shared<HingeEnergy>(mesh.x, idxs[0], idxs[1], idxs[2], idxs[3]));

            //Eigen::Vector3d v = (mesh->vertices[p.second.second] - mesh->vertices[p.second.first]).cast<double>();
            //solver->energyterms.emplace_back(std::make_shared<IBMSpring>(p.second.first, p.second.second, v, v.norm()));
        }
    }

    std::cout << "Found " << count << " opposite pairs." << std::endl;
    //std::cout << "Should match number of internal edges: " << (mesh->edges.size() - mesh->exterior_edges.size()) << std::endl;

    return energies;
}


