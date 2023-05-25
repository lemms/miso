#pragma once

#include <Eigen/Dense>

namespace miso {

void compute_isocline(
        Eigen::MatrixXd& e0,
        Eigen::MatrixXd& e1,
        Eigen::MatrixXd& en0,
        Eigen::MatrixXd& en1,
        const Eigen::Vector3d& min_isocline_direction,
        const Eigen::MatrixXd& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXd& n);

double isocline_length(
        Eigen::MatrixXd& e0,
        Eigen::MatrixXd& e1);

}
