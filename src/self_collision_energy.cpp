
#include "self_collision_energy.hpp"

#include <array>


std::array<double, 108> vals = {
             -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
              0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
              -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
              0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
              0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
              -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
              0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1
};

Eigen::Matrix<double, 12, 9>
    SelfCollisionEnergy::Dt = Eigen::Matrix<double, 9, 12>(vals.data()).transpose();
