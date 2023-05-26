#include <iostream>

#include <Eigen/Dense>

#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>

#include "solver.h"

int main(int argc, char** argv) {
    float min_temperature = 0.0001f;
    float alpha = 0.9f;
    uint32_t max_iterations = 100;
    uint32_t max_inner_iterations = 100;
    float neighbor_stddev = 0.1f;
    bool verbose = false;

    if (argc != 2) {
        std::cout << "Usage miso <PLY file>" << std::endl;

        return 1;   
    }

    if (verbose) {
        std::cout << "Miso" << std::endl;
        std::cout << "The Minimum Isocline Curve Solver" << std::endl;
    }

    Eigen::MatrixXf v;
    Eigen::MatrixXi f;

    igl::readPLY(argv[1], v, f);

    assert(v.cols() == 3);
    assert(f.cols() == 3);

    if (verbose) {
        std::cout << "Processing mesh " << argv[1] << std::endl;
        std::cout << "Vertices: " << v.rows() << std::endl;
        std::cout << "Faces: " << f.rows() << std::endl;
    }

    Eigen::MatrixXf n;

    igl::per_vertex_normals(v, f, n);

    assert(n.cols() == 3);

    if (verbose) {
        std::cout << "Normals: " << n.rows() << std::endl;
    }

    assert(v.rows() == n.rows());   

    Eigen::Vector3f min_isocline_direction;
    float min_isocline_length = std::numeric_limits<float>::infinity();

    miso::solve_min_isocline(
            min_isocline_direction,
            min_isocline_length,
            v,
            f,
            n,
            min_temperature,
            alpha,
            max_iterations,
            max_inner_iterations,
            neighbor_stddev,
            verbose);

    if (verbose) {
        std::cout << "Min isocline curve direction:" << std::endl;
    }

    std::cout << "["
        << min_isocline_direction[0] << ", "
        << min_isocline_direction[1] << ", "
        << min_isocline_direction[2] << "]" << std::endl;

    if (verbose) {
        std::cout << "Min isocline curve length:" << std::endl;
    }

    std::cout << min_isocline_length << std::endl;

    return 0;
}
