// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <iostream>
#include <numeric>
#include <stack>
#include <vector>


class BV {
public:
    BV() : data_index(-1) { };

    Eigen::AlignedBox3d aabb;
    
    // -1 = inactive node, 0 = internal, > 0 = leaf
    int data_index; 
};


class BVH {
    using AABB_Vector = std::vector<Eigen::AlignedBox3d, Eigen::aligned_allocator<Eigen::AlignedBox3d>>;

public:
    bool empty() const { return bvs_.empty(); }

    const BV& root() const { return bvs_.front(); }

    void init(const Eigen::MatrixXi& F, const Eigen::VectorXd& x, double h=0.0) {
        // Each triangle is a leaf node, so assuming a full well-balanced
        // tree, we need at most this many nodes to hold the tree
        // (some nodes will be inactive)
        const int L = std::ceil(std::log(F.rows()) / std::log(2));
        bvs_.resize(std::pow(2, L + 1) - 1);
        //bvs_.resize(std::ceil((4 * F.rows() - 1) / 3));
        std::cout << "Max levels: " << (L+1) << std::endl;
        std::cout << "Nbr BVs: " << bvs_.size() << std::endl;
        std::cout << "Nbr elements: " << F.rows() << std::endl;

        // To begin, construct a BV around all faces, then recurse down
        std::vector<int> idxs(F.rows());
        std::iota(idxs.begin(), idxs.end(), 0);

        AABB_Vector aabbs;
        for (int i=0; i<F.rows(); i++) {
            aabbs.push_back(Eigen::AlignedBox3d());
            for (int j=0; j<3; j++) {
                aabbs.back().extend(x.segment<3>(3*F(i,j)));
            }

            aabbs.back().extend(aabbs.back().min() - Eigen::Vector3d::Ones() * h);
            aabbs.back().extend(aabbs.back().max() + Eigen::Vector3d::Ones() * h);
        }

        build(0, aabbs, idxs);
    }

    void build(int idx, const AABB_Vector& aabbs, const std::vector<int>& idxs, int h=0) {
        if (idx >= bvs_.size()) std::cout << "uh-oh!!! " << idx << std::endl;

        // -1 = inactive node, 0 = internal, > 0 = leaf
        bvs_[idx].data_index = idxs.size() == 1 ? idxs.front() + 1 : 0;

        // Step 1: Construct a BV around all idxs at the requested index
        bvs_[idx].aabb.setEmpty();
        for (int i : idxs) {
            bvs_[idx].aabb.extend(aabbs[i]);
        }

        // Leaf node? Don't recurse anymore
        if (idxs.size() == 1) {
            return;
        }


        // Step 2: Split idxs up into lists
        //auto sizes = bvs_[idx].sizes();
        //size_t split_idx = *std::max_element(sizes.data(), sizes.data()+3);

        // TODO: Redo this...
        std::vector<int>  left_idxs;
        std::vector<int> right_idxs;

        for (size_t i=0; i<idxs.size()/2; i++) {
            left_idxs.push_back(idxs[i]);
        }
        for (size_t i=idxs.size()/2; i<idxs.size(); i++) {
            right_idxs.push_back(idxs[i]);
        }

        // Step 3: Recursively construct hierarchy
        if (!left_idxs.empty())
            build(2 * idx + 1, aabbs, left_idxs, h+1);

        if (!right_idxs.empty())
            build(2 * idx + 2, aabbs, right_idxs, h+1);
    }

    template <typename F>
    void visit(const Eigen::Vector3d& pt, F func) const {
        std::stack<int> q;
        q.push(0); // Start with root node

        while (!q.empty()) {
            int idx = q.top();
            q.pop();

            if (bvs_[idx].aabb.contains(pt)) {
                // Leaf node?
                if (bvs_[idx].data_index > 0) {
                    func(bvs_[idx].data_index - 1);
                }
                // Valid node?
                else if (bvs_[idx].data_index != -1) {
                    for (int i=0; i<2; i++) {
                        q.push(2 * idx + 1 + i);
                    }
                }
            }
        }
    }
    
    void refit(const Eigen::MatrixXi& F, const Eigen::VectorXd& x, double h) {

        // Find first leaf node
        auto it = std::find_if(bvs_.begin(), bvs_.end(),
                                [&](const BV& bv) {
                                    return bv.data_index > 0;
                                });

        if (it == bvs_.end()) {
            return;
        }

        size_t first_leaf = std::distance(bvs_.begin(), it);

        // Refit leaves in parallel
        #pragma omp parallel for
        for (size_t i=first_leaf; i<bvs_.size(); i++) {
            refit(F, x, h, int(i));
        }

        // Not in parallel, refit internal nodes. make sure we skip
        // leaf nodes when we recurse down to them
        refit_internal_nodes();
    }

    bool refit(const Eigen::MatrixXi& F, const Eigen::VectorXd& x, double h, int idx) {
        if (bvs_[idx].data_index == -1) {
            return false;
        }

        bvs_[idx].aabb.setEmpty();

        if (bvs_[idx].data_index > 0) {
            // Refit around face (data_index-1)
            for (int i=0; i<3; i++) {
                bvs_[idx].aabb.extend(x.segment<3>(3*F(bvs_[idx].data_index-1,i)));
            }

            bvs_[idx].aabb.extend(bvs_[idx].aabb.min() - Eigen::Vector3d::Ones() * h);
            bvs_[idx].aabb.extend(bvs_[idx].aabb.max() + Eigen::Vector3d::Ones() * h);
        } else {
            for (int i=0; i<2; ++i) {
                if (refit(F, x, h, 2 * idx + 1 + i)) {
                    bvs_[idx].aabb.extend(bvs_[2*idx+1+i].aabb);
                }
            }
        }

        return true;
    }

protected:
    bool refit_internal_nodes(int idx=0) {
        if (bvs_[idx].data_index == -1) {
            return false;
        }

        if (bvs_[idx].data_index > 0) {
            return true;
        }

        bvs_[idx].aabb.setEmpty();

        for (int i=0; i<2; ++i) {
            if (refit_internal_nodes(2 * idx + 1 + i)) {
                bvs_[idx].aabb.extend(bvs_[2*idx+1+i].aabb);
            }
        }

        return true;
    }

    /*
    void refit_leaf_node(const Eigen::MatrixXi& F,
                         const Eigen::VectorXd& x,
                         double h, BV& bv) {
        bv.aabb.setEmpty();
        for (int i=0; i<3; i++) {
            bv.aabb.extend(x.segment<3>(3 * F(bv.data_index-1,i)));
        }

        bv.aabb.extend(bv.aabb.min() - Eigen::Vector3d::Ones() * h);
        bv.aabb.extend(bv.aabb.max() + Eigen::Vector3d::Ones() * h);
    }

    void refit_internal_node(int idx, BV& bv) {
        bv.aabb.setEmpty();
        for (int i=0; i<2; i++) {
            bv.aabb.extend(bvs_[2*idx+1+i].aabb);
        }
    }
    */

protected:
    std::vector<BV> bvs_;
};
