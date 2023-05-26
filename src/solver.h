#pragma once

#include <Eigen/Dense>

namespace miso {

void solve_min_isocline(
        Eigen::Vector3f& min_isocline_direction,
        float& min_isocline_length,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n,
        const float min_temperature = 0.0001f,
        const float alpha = 0.9f,
        const uint32_t max_iterations = 100,
        const uint32_t max_inner_iterations = 100,
        const float neighbor_stddev = 0.1f,
        const bool verbose = false);

}
