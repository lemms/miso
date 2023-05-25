#pragma once

#include <Eigen/Dense>

namespace miso {

void solve_min_isocline(
        Eigen::Vector3d& min_isocline_direction,
        double& min_isocline_length,
        const Eigen::MatrixXd& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXd& n,
        const double min_temperature = 0.0001,
        const double alpha = 0.9,
        const uint32_t max_iterations = 100,
        const uint32_t max_inner_iterations = 100,
        const double neighbor_stddev = 0.1);

}
