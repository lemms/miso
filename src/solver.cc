#include "solver.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include <Eigen/Dense>

#include "isocline.h"

#define MISO_SOLVER_DEBUG 0
#if MISO_SOLVER_DEBUG
#include <algorithm>
#endif

#define MISO_SOLVER_DEBUG_FILES 0
#if MISO_SOLVER_DEBUG_FILES
#include <filesystem>
#include <fstream>
#endif

namespace miso {

void spherical_coords_to_direction(
        const double theta,
        const double phi,
        Eigen::Vector3d& direction) {

    direction[0] = sin(phi) * cos(theta);
    direction[1] = sin(phi) * sin(theta);
    direction[2] = cos(phi);
}

void random_spherical_coords(
        std::mt19937& gen,
        std::uniform_real_distribution<>& dis,
        double& theta,
        double& phi) {

    theta = 2.0 * M_PI * dis(gen);
    phi = M_PI * dis(gen);
}

void neighbor_spherical_coords(
        std::mt19937& gen,
        std::normal_distribution<>& normal_dis,
        const double theta,
        const double phi,
        double& neighbor_theta,
        double& neighbor_phi) {

    neighbor_theta = std::max(0.0, std::min(2.0 * M_PI, theta + normal_dis(gen)));
    neighbor_phi = std::max(0.0, std::min(M_PI, phi + normal_dis(gen)));
}

void solve_min_isocline(
        Eigen::Vector3d& min_isocline_direction,
        double& min_isocline_length,
        const Eigen::MatrixXd& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXd& n,
        const double min_temperature,
        const double alpha,
        const uint32_t max_iterations,
        const uint32_t max_inner_iterations,
        const double neighbor_stddev) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> normal_dis(0.0, neighbor_stddev);

    min_isocline_length = std::numeric_limits<double>::infinity();

    double min_theta = 0.0;
    double min_phi = 0.0;

    random_spherical_coords(gen, dis, min_theta, min_phi);

    spherical_coords_to_direction(min_theta, min_phi, min_isocline_direction);

    double current_isocline_length = min_isocline_length;
    double current_theta = min_theta;
    double current_phi = min_phi;
    Eigen::Vector3d current_isocline_direction = min_isocline_direction;

    double temperature = 10.0f;
    int32_t iterations = 0;

    bool initialized = false;

    Eigen::MatrixXd min_e0(0, 3);
    Eigen::MatrixXd min_e1(0, 3);
    Eigen::MatrixXd min_en0(0, 3);
    Eigen::MatrixXd min_en1(0, 3);

    Eigen::MatrixXd e0(0, 3);
    Eigen::MatrixXd e1(0, 3);
    Eigen::MatrixXd en0(0, 3);
    Eigen::MatrixXd en1(0, 3);

    while (temperature > min_temperature && iterations < max_iterations) {

        std::cout << "Temperature: " << temperature << std::endl;

        for (uint32_t i = 0; i < max_inner_iterations; ++i) {

            e0.conservativeResize(0, 3);
            e1.conservativeResize(0, 3);
            en0.conservativeResize(0, 3);
            en1.conservativeResize(0, 3);

            double new_theta = 0.0;
            double new_phi = 0.0;

            neighbor_spherical_coords(
                    gen,
                    normal_dis,
                    current_theta,
                    current_phi,
                    new_theta,
                    new_phi);

#if MISO_SOLVER_DEBUG
            std::cout << "Current theta: " << current_theta << std::endl;
            std::cout << "Current phi: " << current_phi << std::endl;
            std::cout << "New theta: " << new_theta << std::endl;
            std::cout << "New phi: " << new_phi << std::endl;
#endif

            Eigen::Vector3d new_isocline_direction;
            
            spherical_coords_to_direction(new_theta, new_phi, new_isocline_direction);


#if MISO_SOLVER_DEBUG
            std::cout << "Search direction: ["
                << new_isocline_direction[0] << ", "
                << new_isocline_direction[1] << ", "
                << new_isocline_direction[2] << "]" << std::endl;
#endif

            compute_isocline(
                    e0,
                    e1,
                    en0,
                    en1,
                    new_isocline_direction,
                    v,
                    f,
                    n);

            assert(e0.rows() == e1.rows());

#if MISO_SOLVER_DEBUG
            std::cout << "Number of edges in isocline: " << e0.rows() << std::endl;
#endif

            const double new_isocline_length = isocline_length(e0, e1);

#if MISO_SOLVER_DEBUG
            std::cout << "Isocline length: " << new_isocline_length << std::endl;
#endif

            if (!initialized) {
                current_theta = new_theta;
                current_phi = new_phi;
                current_isocline_direction = new_isocline_direction;
                current_isocline_length = new_isocline_length;

                initialized = true;
                continue;
            }

            if (current_isocline_length < min_isocline_length) {
                min_isocline_length = current_isocline_length;
                min_theta = current_theta;
                min_phi = current_phi;
                min_isocline_direction = current_isocline_direction;
                min_e0 = e0;
                min_e1 = e1;
                min_en0 = en0;
                min_en1 = en1;
            }

#if MISO_SOLVER_DEBUG
            std::cout << "Current isocline length: " << current_isocline_length << std::endl;
#endif

            double acceptance = exp((current_isocline_length - new_isocline_length) / temperature);

#if MISO_SOLVER_DEBUG
            std::cout << "Acceptance: " << acceptance << std::endl;
#endif

            if (acceptance > dis(gen)) {
#if MISO_SOLVER_DEBUG
                std::cout << "Updating current length" << std::endl;
#endif
                current_theta = new_theta;
                current_phi = new_phi;
                current_isocline_direction = new_isocline_direction;
                current_isocline_length = new_isocline_length;
            }
        }

        // Multiplicative cooling
        temperature *= alpha;
    }

#if MISO_SOLVER_DEBUG_FILES
    // Compute mean edge location
    Eigen::Vector3d mean_location;

    for (size_t e = 0; e < min_e0.rows(); ++e) {
        mean_location += min_e0.row(e);
        mean_location += min_e1.row(e);
    }

    mean_location /= min_e0.rows() * 2.0;

    double edge_distance = 0.0;

    for (size_t e = 0; e < min_e0.rows(); ++e) {
        edge_distance = std::max(edge_distance, (min_e0.row(e).transpose() - mean_location).norm());
        edge_distance = std::max(edge_distance, (min_e1.row(e).transpose() - mean_location).norm());
    }

    double mean_edge_length = 0.0;

    for (size_t e = 0; e < min_e0.rows(); ++e) {
        mean_edge_length += (min_e1.row(e) - min_e0.row(e)).norm();
    }

    mean_edge_length /= min_e0.rows();

    {
        std::filesystem::path dir_obj_path = std::filesystem::temp_directory_path() / "miso_debug_direction.obj";

        std::cout << "Writing debug dir OBJ to " << dir_obj_path << std::endl;

        std::ofstream dir_obj(dir_obj_path);
        dir_obj << "v " << mean_location[0] << " " << mean_location[1] << " " << mean_location[2] << std::endl;
        dir_obj << "v " << mean_location[0] + min_isocline_direction[0] * edge_distance << " " << mean_location[1] + min_isocline_direction[1] * edge_distance << " " << mean_location[2] + min_isocline_direction[2] * edge_distance << std::endl;

        dir_obj << "l 1 2" << std::endl;

        dir_obj.close();
    }

    {
        std::filesystem::path edge_obj_path = std::filesystem::temp_directory_path() / "miso_debug_points.obj";

        std::cout << "Writing debug edge OBJ to " << edge_obj_path << std::endl;

        std::ofstream edge_obj(edge_obj_path);
        for (size_t e = 0; e < min_e0.rows(); ++e) {
            edge_obj << "v " << min_e0(e, 0) << " " << min_e0(e, 1) << " " << min_e0(e, 2) << std::endl;
            edge_obj << "v " << min_e1(e, 0) << " " << min_e1(e, 1) << " " << min_e1(e, 2) << std::endl;
        }

        for (size_t e = 0; e < min_e0.rows(); ++e) {
            edge_obj << "l " << e * 2 + 1 << " " << e * 2 + 2 << std::endl;
        }

        edge_obj.close();
    }

    {
        std::filesystem::path normals_obj_path = std::filesystem::temp_directory_path() / "miso_debug_normals.obj";

        std::cout << "Writing debug edge OBJ to " << normals_obj_path << std::endl;

        std::ofstream normals_obj(normals_obj_path);
        for (size_t e = 0; e < min_e0.rows(); ++e) {
            normals_obj << "v " << min_e0(e, 0) << " " << min_e0(e, 1) << " " << min_e0(e, 2) << std::endl;
            normals_obj << "v " << min_e0(e, 0) + min_en0(e, 0) * mean_edge_length << " " << min_e0(e, 1) + min_en0(e, 1) * mean_edge_length << " " << min_e0(e, 2) + min_en0(e, 2) * mean_edge_length << std::endl;
            normals_obj << "v " << min_e1(e, 0) << " " << min_e1(e, 1) << " " << min_e1(e, 2) << std::endl;
            normals_obj << "v " << min_e1(e, 0) + min_en1(e, 0) * mean_edge_length << " " << min_e1(e, 1) + min_en1(e, 1) * mean_edge_length << " " << min_e1(e, 2) + min_en1(e, 2) * mean_edge_length << std::endl;
        }

        for (size_t e = 0; e < min_e0.rows(); ++e) {
            normals_obj << "l " << e * 4 + 1 << " " << e * 4 + 2 << std::endl;
            normals_obj << "l " << e * 4 + 3 << " " << e * 4 + 4 << std::endl;
        }

        normals_obj.close();
    }
#endif
}

}
