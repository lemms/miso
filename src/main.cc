#include <iostream>
#include <string>

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

    std::vector<std::string> arguments(argv + 1, argv + argc);

    if (arguments.empty()) {
        std::cout << "Usage miso <PLY file> --min-temp=<min temperature> --alpha=<alpha> --max-iter=<max iterations> --max-inner-iter=<max inner iterations> --neighbor-stddev=<neighbor standard deviation> --verbose" << std::endl;

        return 0;
    }

    std::string ply_file_name;

    for (const std::string& arg : arguments) {
        const auto n = arg.find('=');
        if (n == std::string::npos) {
            if (arg == "--verbose") {
                verbose = true;
            } else {
                ply_file_name = arg;
            }
        } else {
            std::string key = arg.substr(0, n);
            std::string value = arg.substr(n + 1, std::string::npos);
            if (key == "--min-temp") {
                min_temperature = std::stof(value);
            } else if (key == "--alpha") {
                alpha = std::stof(value);
            } else if (key == "--max-iter") {
                max_iterations = std::stoi(value);
            } else if (key == "--max-inner-iter") {
                max_inner_iterations = std::stoi(value);
            } else if (key == "--neighbor-stddev") {
                neighbor_stddev = std::stof(value);
            }
        }
    }

    if (verbose) {
        std::cout << "Miso" << std::endl;
        std::cout << "The Minimum Isocline Curve Solver" << std::endl;
    }

    Eigen::MatrixXf v;
    Eigen::MatrixXi f;

    igl::readPLY(ply_file_name.c_str(), v, f);

    assert(v.cols() == 3);
    assert(f.cols() == 3);

    if (verbose) {
        std::cout << "Processing mesh " << ply_file_name << std::endl;
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
