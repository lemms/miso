#include "isocline.h"

#include <Eigen/Dense>

#define MISO_ISOCLINE_DEBUG 0
#if MISO_ISOCLINE_DEBUG
#include <iostream>
#endif

namespace miso {

void compute_isocline(
        Eigen::MatrixXd& e0,
        Eigen::MatrixXd& e1,
        Eigen::MatrixXd& en0,
        Eigen::MatrixXd& en1,
        const Eigen::Vector3d& min_isocline_direction,
        const Eigen::MatrixXd& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXd& n) {

    for (size_t face = 0; face < f.rows(); ++face) {
        int isocline_index = 0;
        for (size_t edge = 0; edge < 3; ++edge) {
            Eigen::Vector3d n0 = n.row(f(face, edge));
            Eigen::Vector3d n1 = n.row(f(face, (edge + 1) % 3));

            double a = -n0.dot(min_isocline_direction) / (n1.dot(min_isocline_direction) - n0.dot(min_isocline_direction));           

            if (a >= 0.0 && a <= 1.0) {
                Eigen::Vector3d v0 = v.row(f(face, edge));
                Eigen::Vector3d v1 = v.row(f(face, (edge + 1) % 3));

                Eigen::Vector3d isocline_vertex = (1.0 - a) * v0 + a * v1;
                Eigen::Vector3d isocline_normal = (1.0 - a) * n0 + a * n1;

                // Min isocline curve passes through edge
                if (isocline_index == 0) {
                    e0.conservativeResize(e0.rows() + 1, e0.cols());
                    e0.row(e0.rows() - 1) = isocline_vertex;

                    en0.conservativeResize(en0.rows() + 1, en0.cols());
                    en0.row(en0.rows() - 1) = isocline_normal;
                } else {
                    e1.conservativeResize(e1.rows() + 1, e1.cols());
                    e1.row(e1.rows() - 1) = isocline_vertex;

                    en1.conservativeResize(en1.rows() + 1, en1.cols());
                    en1.row(en1.rows() - 1) = isocline_normal;
                }

                ++isocline_index;
            }
        }

        if (isocline_index == 1) {
            // The isocline curve is degenerate on this face

#if MISO_ISOCLINE_DEBUG
            std::cerr << "Warning: Isocline curve is degenerate on face " << face << std::endl;
#endif

            e1.conservativeResize(e1.rows() + 1, e1.cols());
            e1.row(e1.rows() - 1) = e0.row(e1.rows() - 1);

            en1.conservativeResize(en1.rows() + 1, en1.cols());
            en1.row(en1.rows() - 1) = en0.row(en1.rows() - 1);
        }

        assert(e0.rows() == e1.rows());
        assert(en0.rows() == en1.rows());
    }
}

double isocline_length(
        Eigen::MatrixXd& e0,
        Eigen::MatrixXd& e1) {

    assert(e0.rows() == e1.rows());

    return (e1 - e0).rowwise().norm().sum();
}

}
