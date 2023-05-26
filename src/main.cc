#include <iostream>
#include <string>

#include <Eigen/Dense>

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/readSTL.h>
#include <igl/per_vertex_normals.h>

#include "solver.h"

int main(int argc, char** argv) {
    float angle = 0.0f;
    float min_temperature = 0.0001f;
    float alpha = 0.9f;
    uint32_t max_iterations = 100;
    uint32_t max_inner_iterations = 100;
    float neighbor_stddev = 0.1f;
    bool verbose = false;
    bool debug_files = false;

    std::vector<std::string> arguments(argv + 1, argv + argc);

    if (arguments.empty()) {
        std::cout << "Usage miso <mesh file> --angle=<isocline angle> --min-temp=<min temperature> --alpha=<alpha> --max-iter=<max iterations> --max-inner-iter=<max inner iterations> --neighbor-stddev=<neighbor standard deviation> --verbose --debug-files" << std::endl;

        return 0;
    }

    std::string mesh_file_name;

    for (const std::string& arg : arguments) {
        const auto n = arg.find('=');
        if (n == std::string::npos) {
            if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--debug-files") {
                debug_files = true;
            } else {
                mesh_file_name = arg;
            }
        } else {
            std::string key = arg.substr(0, n);
            std::string value = arg.substr(n + 1, std::string::npos);
            if (key == "--angle") {
                angle = std::stof(value);
            } else if (key == "--min-temp") {
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

    std::filesystem::path mesh_file_path(mesh_file_name);

    Eigen::MatrixXf n;

    if (mesh_file_path.extension() == ".obj") {
        igl::readOBJ(mesh_file_name.c_str(), v, f);
        igl::per_vertex_normals(v, f, n);
    } else if (mesh_file_path.extension() == ".off") {
        igl::readOFF(mesh_file_name.c_str(), v, f);
        igl::per_vertex_normals(v, f, n);
    } else if (mesh_file_path.extension() == ".ply") {
        igl::readPLY(mesh_file_name.c_str(), v, f);
        igl::per_vertex_normals(v, f, n);
    } else if (mesh_file_path.extension() == ".stl") {
        std::ifstream mesh_file(mesh_file_name);
        igl::readSTL(mesh_file, v, f, n);
    } else {
        std::cerr << "Unsupported mesh file format: " << mesh_file_path.extension() << std::endl;

        return 1;
    }

    assert(v.cols() == 3);
    assert(n.cols() == 3);

    if (f.cols() != 3) {
        std::cerr << "Input file contains non-triangular faces. Miso expects triangle mesh input." << std::endl;

        return 1;
    }

    if (verbose) {
        std::cout << "Isocline angle: " << angle << std::endl;
        std::cout << "Processing mesh: " << mesh_file_name << std::endl;
        std::cout << "Vertices: " << v.rows() << std::endl;
        std::cout << "Faces: " << f.rows() << std::endl;
        std::cout << "Normals: " << n.rows() << std::endl;
    }

    assert(v.rows() == n.rows());   

    Eigen::Vector3f min_isocline_direction;
    float min_isocline_length = std::numeric_limits<float>::infinity();

    miso::solve_min_isocline(
            min_isocline_direction,
            min_isocline_length,
            angle,
            v,
            f,
            n,
            min_temperature,
            alpha,
            max_iterations,
            max_inner_iterations,
            neighbor_stddev,
            verbose,
            debug_files);

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
