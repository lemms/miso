#include <iostream>

#include <Eigen/Dense>

#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>

#include "solver.h"

int main(int argc, char** argv) {
    std::cout << "Miso" << std::endl;
    std::cout << "The Minimum Isocline Curve Solver" << std::endl;

    if (argc != 2) {
        std::cout << "Usage miso <PLY file>" << std::endl;

        return 1;   
    }

    Eigen::MatrixXd v;
    Eigen::MatrixXi f;

    igl::readPLY(argv[1], v, f);

    assert(v.cols() == 3);
    assert(f.cols() == 3);

    std::cout << "Processing mesh " << argv[1] << std::endl;
    std::cout << "Vertices: " << v.rows() << std::endl;
    std::cout << "Faces: " << f.rows() << std::endl;

    Eigen::MatrixXd n;

    igl::per_vertex_normals(v, f, n);

    assert(n.cols() == 3);

    std::cout << "Normals: " << n.rows() << std::endl;

    assert(v.rows() == n.rows());   

    Eigen::Vector3d min_isocline_direction;
    double min_isocline_length = std::numeric_limits<double>::infinity();

    miso::solve_min_isocline(
            min_isocline_direction,
            min_isocline_length,
            v, f, n);

    std::cout << "Min isocline curve direction: ["
        << min_isocline_direction[0] << ", "
        << min_isocline_direction[1] << ", "
        << min_isocline_direction[2] << "]" << std::endl;

    std::cout << "Min isocline curve length: "
        << min_isocline_length << std::endl;

    return 0;
}
