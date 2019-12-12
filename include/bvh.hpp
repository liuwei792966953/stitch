// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <array>
#include <iostream>
#include "kdop.hpp"
#include <numeric>
#include <stack>
#include <unordered_set>
#include <vector>



template <typename Type>
class BoundingVolume : public Type  {
public:
    BoundingVolume() : Type(), data_index(-1) { };
    
    // -1 = inactive node, 0 = internal, > 0 = leaf
    int data_index; 
};


// N = branching factor
template<size_t N>
class BVH {
public:
    using BV = BoundingVolume<kDOP26d>;

    void init(const Eigen::MatrixXi& F, const Eigen::VectorXd& x, double h=0.0) {
        // Each triangle is a leaf node, so assuming a full well-balanced
        // tree, we need at most this many nodes to hold the tree
        // (some nodes will be inactive)
        height_ = std::ceil(std::log(F.rows()) / std::log(N)) + 1;
        bvs_.resize(std::pow(N, height_) - 1);
        std::cout << F.rows() << "; " << bvs_.size() << "; " << height_ << std::endl;

        // To begin, construct a BV around all faces, then recurse down
        std::vector<BV> leaves(F.rows());
        for (int i=0; i<F.rows(); i++) {
            leaves[i].data_index = i + 1;
            for (int j=0; j<3; j++) {
                leaves[i].extend(x.segment<3>(3*F(i,j)));
            }
            leaves[i].extendK(leaves[i].min() - BV::VectorK::Constant(h));
            leaves[i].extendK(leaves[i].max() + BV::VectorK::Constant(h));
        }

        build(0, leaves);

        // Assign each vertex to a "representative triangle". It can only belong
        // to one
        std::unordered_set<int> assigned;

        rep_tri_vertices_.resize(F.rows(), { -1, -1, -1 }); 
        for (int i=0; i<F.rows(); i++) {
            // Find first available slot for this face
            size_t idx = 0;
            for (int j=0; j<3; j++) {
                int vidx = F(i,j);
                if (!assigned.count(vidx)) {
                    rep_tri_vertices_[i][idx++] = vidx;
                    assigned.insert(vidx);
                }
            }
        }
    }

    void build(int idx, const std::vector<BV>& leaves, int height=0) {
        if (idx >= bvs_.size()) std::cout << "uh-oh!!! " << idx << std::endl;

        // Leaf node? Don't recurse anymore
        if (leaves.size() == 1) {
            bvs_[idx] = leaves.front();
            return;
        }

        bvs_[idx].data_index = 0; // 0 => internal node

        // Step 1: Construct a BV around all leaves at the requested index
        bvs_[idx].setEmpty();
        for (const BV& leaf : leaves) {
            bvs_[idx].extend(leaf);
        }


        // Step 2: Split leaves up into lists
        // TODO: Make the "short" side on the right, then we can resize bvs_
        // and reclaim some space
        int split_idx;
        (bvs_[idx].max() - bvs_[idx].min()).maxCoeff(&split_idx);

        std::vector<std::pair<double, int>> vals;
        for (int i=0; i<leaves.size(); i++) {
            // Sort by midpoint
            const double mid = (leaves[i].min()[split_idx] + leaves[i].max()[split_idx]) * 0.5;
            vals.push_back({ mid, i });
        }

        std::sort(vals.begin(), vals.end(),
                [](const auto& p1, const auto& p2) { return p1.first < p2.first; });

        std::vector<BV> split_idxs[N];

        const size_t group_size = int(std::ceil(double(leaves.size()) / double(N)));
        for (int i=0; i<N; i++) {
            for (size_t j=i*group_size; j<std::min(leaves.size(), (i+1)*group_size); j++) {
                split_idxs[i].push_back(leaves[vals[j].second]);
            }
        }

        if ((height % 2) != 0) { std::swap(split_idxs[0], split_idxs[1]); }
        
        // Step 3: Recursively construct hierarchy
        for (size_t i=0; i<N; i++) {
            if (!split_idxs[i].empty()) {
                build(get_child_index(idx, i), split_idxs[i], height+1);
            }
        }
    }
 
    void refit(const Eigen::MatrixXi& F, const Eigen::VectorXd& x, double h) {
        /*
        size_t idx = bvs_.size() - 1;
        for (auto it=bvs_.rbegin(); it!=bvs_.rend(); ++it) {
            if (it->data_index != -1) {
                it->setEmpty();

                if (it->data_index > 0) {
                    // Refit around face (data_index-1)
                    for (int j=0; j<3; j++) {
                        it->extend(x.segment<3>(3*F(it->data_index-1,j)));
                    }
                    it->extendK(it->min() - BV::VectorK::Constant(h));
                    it->extendK(it->max() + BV::VectorK::Constant(h));
                } else {
                    // Internal node. Refit around children
                    for (int i=0; i<N; ++i) {
                        if (bvs_[get_child_index(idx, i)].data_index != -1) {
                            it->extend(bvs_[get_child_index(idx, i)]);
                        }
                    }
                }
            }
            idx--;
        }
        */
        for (int l=0; l<height_; l++) {
            // The nodes at depth l are N^l-1 through N^(l+1)-1, and
            // can all be refitted in parallel
            #pragma omp parallel for
            for (int i=std::pow(N, l) - 1; i<std::pow(N, l+1) - 1; i++) {
                refit_node(i, F, x, h);
            }
        }
    }

    template <typename Derived, typename F>
    void visit(const Eigen::MatrixBase<Derived>& pt, F func) const {
        std::stack<int> q;
        q.push(0); // Start with root node

        while (!q.empty()) {
            int idx = q.top();
            q.pop();

            if (bvs_[idx].contains(pt)) {
                // Leaf node?
                if (bvs_[idx].data_index > 0) {
                    func(bvs_[idx].data_index - 1);
                }
                // Valid node?
                else if (bvs_[idx].data_index != -1) {
                    for (int i=0; i<N; i++) {
                        q.push(get_child_index(idx, i));
                    }
                }
            }
        }
    }

    template <typename F>
    void self_intersect(const Eigen::MatrixXi& faces, F f) const {
        auto go = [&](int i1, int i2) {

        std::stack<std::pair<int, int>> s;
        s.push({ i1, i2 });
        //s.push({ 0, 0 });

        while (!s.empty()) {
            auto idxs = s.top();
            s.pop();

            if (bvs_[idxs.first].intersects(bvs_[idxs.second])) {
                int d1 = bvs_[idxs.first].data_index;
                int d2 = bvs_[idxs.second].data_index;

                if (d1 > 0 && d2 > 0) {
                    // Both leaf nodes
                    for (int i=0; i<3; i++) {
                        if (rep_tri_vertices_[d1-1][i] != -1) {
                            f(rep_tri_vertices_[d1-1][i], d2-1);
                        }
                        if (rep_tri_vertices_[d2-1][i] != -1) {
                            f(rep_tri_vertices_[d2-1][i], d1-1);
                        }
                    }
                } else if (d1 > 0 && d2 != -1) {
                    // First is leaf node, other is internal,
                    // so just descend down the second
                    for (int i=0; i<N; i++) {
                        s.push({ idxs.first, get_child_index(idxs.second, i) });
                    }
                } else if (d1 != -1 && d2 > 0) {
                    // First is internal node, other is leaf,
                    // so just descend down the first
                    for (int i=0; i<N; i++) {
                        s.push({ get_child_index(idxs.first, i), idxs.second });
                    }
                } else if (d1 != -1 && d2 != -1) {
                    // Both internal nodes, intersect children pairs
                    for (int i=0; i<N; i++) {
                        for (int j=(idxs.first == idxs.second ? i : 0); j<N; j++) {
                            s.push({ get_child_index(idxs.first, i), get_child_index(idxs.second, j) });
                        }
                    }
                }
            }
        }
        };

        std::array<std::pair<int, int>, N*N> pairs;
        for (size_t i=0; i<N; i++) {
            for (size_t j=0; j<N; j++) {
                pairs[i*N+j] = std::make_pair(1+i, 1+j);
            }
        }

        #pragma omp parallel for
        for (size_t i=0; i<pairs.size(); i++) {
            go(pairs[i].first, pairs[i].second);
        }
    }
   
protected:
    void refit_node(int idx, const Eigen::MatrixXi& F, const Eigen::VectorXd& x, double h) {
        if (bvs_[idx].data_index == -1) {
            return;
        }

        bvs_[idx].setEmpty();

        if (bvs_[idx].data_index > 0) {
            // Refit around face (data_index-1)
            for (int j=0; j<3; j++) {
                bvs_[idx].extend(x.segment<3>(3*F(bvs_[idx].data_index-1, j)));
            }
            bvs_[idx].extendK(bvs_[idx].min() - BV::VectorK::Constant(h));
            bvs_[idx].extendK(bvs_[idx].max() + BV::VectorK::Constant(h));
        } else {
            // Internal node. Refit around children
            for (int i=0; i<N; ++i) {
                if (bvs_[get_child_index(idx, i)].data_index != -1) {
                    bvs_[idx].extend(bvs_[get_child_index(idx, i)]);
                }
            }
        }
    }

    int get_child_index(int idx, int child) const { return N*idx+child+1; }

protected:
    std::vector<BV> bvs_;
    double height_ = 0;

    std::vector<std::array<int, 3>> rep_tri_vertices_;
};
