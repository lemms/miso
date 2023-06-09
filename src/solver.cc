#include "solver.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

#include <Eigen/Dense>

#include "isocline.h"

#define MISO_SOLVER_DEBUG 0
#if MISO_SOLVER_DEBUG
#include <algorithm>
#endif

namespace {
    constexpr float threshold = 1e-7f;
}

namespace miso {

void spherical_coords_to_direction(
        const float theta,
        const float phi,
        Eigen::Vector3f& direction) {

    direction[0] = sinf(phi) * cosf(theta);
    direction[1] = sinf(phi) * sinf(theta);
    direction[2] = cosf(phi);
}

void random_spherical_coords(
        std::mt19937& gen,
        std::uniform_real_distribution<float>& dis,
        float& theta,
        float& phi) {

    theta = 2.0f * static_cast<float>(M_PI) * dis(gen);
    phi = acos(1.0f - 2.0f * dis(gen));
}

void neighbor_spherical_coords(
        std::mt19937& gen,
        std::normal_distribution<float>& normal_dis,
        const float theta,
        const float phi,
        float& neighbor_theta,
        float& neighbor_phi) {

    neighbor_theta = std::max(0.0f, std::min(2.0f * static_cast<float>(M_PI), theta + normal_dis(gen)));
    neighbor_phi = std::max(0.0f, std::min(static_cast<float>(M_PI), phi + normal_dis(gen)));

    // Wrap neighbor coordinates
    while (neighbor_theta < 0.0f) {
        neighbor_theta += 2.0f * static_cast<float>(M_PI);
    }
    while (neighbor_theta > 2.0f * static_cast<float>(M_PI)) {
        neighbor_theta -= 2.0f * static_cast<float>(M_PI);
    }

    while (neighbor_phi < 0.0f) {
        neighbor_phi += static_cast<float>(M_PI);
    }
    while (neighbor_phi > static_cast<float>(M_PI)) {
        neighbor_phi -= static_cast<float>(M_PI);
    }
}

void solve_min_isocline(
        Eigen::Vector3f& min_isocline_direction,
        float& min_isocline_length,
        const float angle,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n,
        const float min_temperature,
        const float alpha,
        const uint32_t max_iterations,
        const uint32_t max_inner_iterations,
        const float neighbor_stddev,
        const bool verbose,
        const bool debug_files) {

    const float cos_angle = std::cos(static_cast<float>(M_PI) * 0.5f - angle);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::normal_distribution<float> normal_dis(0.0f, neighbor_stddev);

    min_isocline_length = std::numeric_limits<float>::infinity();

    float min_theta = 0.0f;
    float min_phi = 0.0f;

    random_spherical_coords(gen, dis, min_theta, min_phi);

    spherical_coords_to_direction(min_theta, min_phi, min_isocline_direction);

    float current_isocline_length = min_isocline_length;
    float current_theta = min_theta;
    float current_phi = min_phi;
    Eigen::Vector3f current_isocline_direction = min_isocline_direction;

    float temperature = 10.0f;
    int32_t iterations = 0;

    bool initialized = false;

    Eigen::MatrixXf min_e0(0, 3);
    Eigen::MatrixXf min_e1(0, 3);
    Eigen::MatrixXf min_en0(0, 3);
    Eigen::MatrixXf min_en1(0, 3);

    Eigen::MatrixXf current_e0(0, 3);
    Eigen::MatrixXf current_e1(0, 3);
    Eigen::MatrixXf current_en0(0, 3);
    Eigen::MatrixXf current_en1(0, 3);

    Eigen::MatrixXf e0(0, 3);
    Eigen::MatrixXf e1(0, 3);
    Eigen::MatrixXf en0(0, 3);
    Eigen::MatrixXf en1(0, 3);

    while (temperature > min_temperature && iterations < max_iterations) {

        if (verbose) {
            std::cout << "Temperature: " << temperature << std::endl;
        }

        for (uint32_t i = 0; i < max_inner_iterations; ++i) {

            e0.conservativeResize(0, 3);
            e1.conservativeResize(0, 3);
            en0.conservativeResize(0, 3);
            en1.conservativeResize(0, 3);

            float new_theta = 0.0f;
            float new_phi = 0.0f;

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

            Eigen::Vector3f new_isocline_direction = {0.0f, 0.0f, 0.0f};
            
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
                    cos_angle,
                    v,
                    f,
                    n);

            assert(e0.rows() == e1.rows());
            assert(e0.rows() == en0.rows());
            assert(e0.rows() == en1.rows());

#if MISO_SOLVER_DEBUG
            std::cout << "Number of edges in isocline: " << e0.rows() << std::endl;
#endif

            if (e0.rows() == 0) {
#if MISO_SOLVER_DEBUG
                std::cout << "No isocline edges found for search direction, skipping candidate search direction." << std::endl;
#endif

                continue;
            }

            const float new_isocline_length = isocline_length(e0, e1);

            assert(!isnan(new_isocline_length));

            if (std::abs(new_isocline_length) < threshold) {
#if MISO_SOLVER_DEBUG
                std::cout << "Isocline curve is degenerate for search direction, skipping candidate search direction." << std::endl;
#endif

                continue;
            }

#if MISO_SOLVER_DEBUG
            std::cout << "Isocline length: " << new_isocline_length << std::endl;
#endif

            if (!initialized) {
                current_theta = new_theta;
                current_phi = new_phi;
                current_isocline_direction = new_isocline_direction;
                current_isocline_length = new_isocline_length;
                current_e0 = e0;
                current_e1 = e1;
                current_en0 = en0;
                current_en1 = en1;

                initialized = true;
                continue;
            }

            if (current_isocline_length < min_isocline_length) {
                min_isocline_length = current_isocline_length;
                min_theta = current_theta;
                min_phi = current_phi;
                min_isocline_direction = current_isocline_direction;
                min_e0 = current_e0;
                min_e1 = current_e1;
                min_en0 = current_en0;
                min_en1 = current_en1;
            }

#if MISO_SOLVER_DEBUG
            std::cout << "Current isocline length: " << current_isocline_length << std::endl;
#endif

            const float acceptance = exp((current_isocline_length - new_isocline_length) / temperature);

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
                current_e0 = e0;
                current_e1 = e1;
                current_en0 = en0;
                current_en1 = en1;
            }
        }

        // Multiplicative cooling
        temperature *= alpha;
    }

    if (debug_files) {
        if (min_e0.rows() == 0) {
            std::cerr << "Failed to find an isocline curve for input mesh." << std::endl;

            return;
        }

        // Compute mean edge location
        Eigen::Vector3f mean_location = {0.0f, 0.0f, 0.0f};

        for (size_t e = 0; e < min_e0.rows(); ++e) {
            mean_location += min_e0.row(e);
            mean_location += min_e1.row(e);
        }

        mean_location /= min_e0.rows() * 2.0f;

        std::cout << "Mean location: " << mean_location[0] << ", " << mean_location[1] << ", " << mean_location[2] << std::endl;

        float edge_distance = 0.0f;

        for (size_t e = 0; e < min_e0.rows(); ++e) {
            edge_distance = std::max(edge_distance, (min_e0.row(e).transpose() - mean_location).norm());
            edge_distance = std::max(edge_distance, (min_e1.row(e).transpose() - mean_location).norm());
        }

        std::cout << "Edge distance: " << edge_distance << std::endl;

        float mean_edge_length = 0.0f;

        for (size_t e = 0; e < min_e0.rows(); ++e) {
            mean_edge_length += (min_e1.row(e) - min_e0.row(e)).norm();
        }

        mean_edge_length /= min_e0.rows();

        {
            std::filesystem::path dir_obj_path = std::filesystem::temp_directory_path() / "miso_debug_direction.obj";

            std::cout << "Writing debug direction OBJ to " << dir_obj_path << std::endl;

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

            std::cout << "Writing debug edge normals OBJ to " << normals_obj_path << std::endl;

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

        {
            std::filesystem::path mesh_normals_obj_path = std::filesystem::temp_directory_path() / "miso_debug_mesh_normals.obj";

            std::cout << "Writing debug mesh normals OBJ to " << mesh_normals_obj_path << std::endl;

            std::ofstream mesh_normals_obj(mesh_normals_obj_path);
            for (size_t i = 0; i < n.rows(); ++i) {
                if (isnan(n(i, 0)) || isnan(n(i, 1)) || isnan(n(i, 2))) {
                    continue;
                }

                mesh_normals_obj << "v " << v(i, 0) << " " << v(i, 1) << " " << v(i, 2) << std::endl;
                mesh_normals_obj << "v " << v(i, 0) + n(i, 0) * mean_edge_length << " " << v(i, 1) + n(i, 1) * mean_edge_length << " " << v(i, 2) + n(i, 2) * mean_edge_length << std::endl;
            }

            size_t idx = 0;
            for (size_t i = 0; i < n.rows(); ++i) {
                if (isnan(n(i, 0)) || isnan(n(i, 1)) || isnan(n(i, 2))) {
                    continue;
                }

                mesh_normals_obj << "l " << idx * 2 + 1 << " " << idx * 2 + 2 << std::endl;
                ++idx;
            }

            mesh_normals_obj.close();
        }
    }
}

}
