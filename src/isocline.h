#pragma once

#include <Eigen/Dense>

namespace miso {

void compute_isocline(
        Eigen::MatrixXf& e0,
        Eigen::MatrixXf& e1,
        Eigen::MatrixXf& en0,
        Eigen::MatrixXf& en1,
        const Eigen::Vector3f& min_isocline_direction,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n);

float isocline_length(
        Eigen::MatrixXf& e0,
        Eigen::MatrixXf& e1);

}
