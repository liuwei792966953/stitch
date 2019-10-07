// Copyright (C) 2019 David Harmon and Artificial Necessity
// This code distributed under zlib, see LICENSE.txt for terms.

#pragma once

#include <Eigen/Core>


template<typename Scalar, int K>
class kDOP {
public:
    using VectorK  = Eigen::Matrix<Scalar, K/2, 1>;
    using MatrixK3 = Eigen::Matrix<Scalar, K/2, 3>;

    kDOP() { setEmpty(); }

    void setEmpty() {
        min_ = VectorK::Constant(std::numeric_limits<Scalar>::max());
        max_ = VectorK::Constant(std::numeric_limits<Scalar>::min());
    }

    template<typename Derived>
    kDOP& extendK(const Eigen::MatrixBase<Derived>& vals) {
        min_ = min_.cwiseMin(vals);
        max_ = max_.cwiseMax(vals);
        return *this;
    }

    template<typename Derived>
    kDOP& extend(const Eigen::MatrixBase<Derived>& p) {
        VectorK p_proj = project(p);
        min_ = min_.cwiseMin(p_proj);
        max_ = max_.cwiseMax(p_proj);
        return *this;
    }

    kDOP& extend(const kDOP& b) {
        min_ = min_.cwiseMin(b.min_);
        max_ = max_.cwiseMax(b.max_);
        return *this;
    }

    template<typename DerivedA, typename DerivedB>
    kDOP& extend(const Eigen::MatrixBase<DerivedA>& x,
                 const Eigen::MatrixBase<DerivedB>& idxs) {
        for (int i=0; i<idxs.size(); i++) {
            extend(x.template segment<3>(3*idxs[i]));
        }
        return *this;
    }

    template<typename Derived>
    bool contains(const Eigen::MatrixBase<Derived>& p) const {
        VectorK p_proj = project(p);
        return (min_.array() <= p_proj.array()).all() &&
               (p_proj.array() <= max_.array()).all();
    }

    bool intersects(const kDOP& b) const {
        return (min_.array() <= (b.max)().array()).all() &&
                ((b.min)().array() <= max_.array()).all();
    }

    template<typename Derived>
    VectorK project(const Eigen::MatrixBase<Derived>& p) const {
        return Eigen::Map<const MatrixK3>(dirs.data()) * p;
    }

    const VectorK& min() const { return min_; }
    const VectorK& max() const { return max_; }

protected:
    VectorK min_;
    VectorK max_;

    static constexpr std::array<Scalar, 39> dirs = {
            1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
            0, 1, 0, 1, -1, 1, -1, 1, 0, 1, -1, 0, 1,
            0, 0, 1, 1, 1, -1, -1, 0, 1, 1, 0, -1, -1 };
};

using kDOP26d = kDOP<double, 26>;

