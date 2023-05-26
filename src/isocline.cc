#include "isocline.h"

#include <algorithm>
#include <thread>
#include <tuple>

#include <Eigen/Dense>

#define MISO_ISOCLINE_DEBUG 0
#if MISO_ISOCLINE_DEBUG
#include <iostream>
#endif

namespace miso {

void gather_edges(
        std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>>& edges,
        const Eigen::Vector3f& min_isocline_direction,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n,
        const size_t start_face,
        const size_t end_face) {

    for (size_t face = start_face; face < end_face; ++face) {
        uint8_t isocline_index = 0;

        std::array<float, 3> n_dot_d{0, 0, 0};

        for (size_t edge = 0; edge < 3; ++edge) {
            n_dot_d[edge] = n.row(f(face, edge)).dot(min_isocline_direction);
        }

        for (size_t edge = 0; edge < 3; ++edge) {
            const uint32_t e0 = edge;
            const uint32_t e1 = (edge + 1) % 3;

            const uint32_t i0 = f(face, e0);
            const uint32_t i1 = f(face, e1);

            const Eigen::Vector3f& n0 = n.row(i0);
            const Eigen::Vector3f& n1 = n.row(i1);

            const float a = -n_dot_d[e0] / (n_dot_d[e1] - n_dot_d[e0]);

            if (a < 0.0 || a > 1.0) {
                continue;
            }

            const Eigen::Vector3f& v0 = v.row(i0);
            const Eigen::Vector3f& v1 = v.row(i1);


            const Eigen::Vector3f isocline_vertex = (1.0f - a) * v0 + a * v1;
            const Eigen::Vector3f isocline_normal = (1.0f - a) * n0 + a * n1;

            // Min isocline curve passes through edge
            if (isocline_index == 0) {
                edges.emplace_back(isocline_vertex, isocline_normal, isocline_vertex, isocline_normal);
            } else if (isocline_index == 1) {
                std::get<2>(edges[edges.size() - 1]) = isocline_vertex;
                std::get<3>(edges[edges.size() - 1]) = isocline_normal;
                break;
            }

            ++isocline_index;
        }

#if MISO_ISOCLINE_DEBUG
        if (isocline_index == 1) {
            // The isocline curve is degenerate on this face

            std::cerr << "Warning: Isocline curve is degenerate on face " << face << std::endl;
        }
#endif
    }
}

void compute_isocline(
        Eigen::MatrixXf& e0,
        Eigen::MatrixXf& e1,
        Eigen::MatrixXf& en0,
        Eigen::MatrixXf& en1,
        const Eigen::Vector3f& min_isocline_direction,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n) {

    const auto processor_count = std::thread::hardware_concurrency();

#if MISO_ISOCLINE_DEBUG
    std::cout << "Processor count: " << processor_count << std::endl;
#endif

    std::vector<std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>>> thread_edges(processor_count);

    size_t num_faces_per_thread = f.rows() / processor_count;

#if MISO_ISOCLINE_DEBUG
    std::cout << "Faces per thread: " << num_faces_per_thread << std::endl;
#endif

    std::vector<std::thread> threads;

    for (size_t i = 0; i < processor_count; ++i) {
        threads.push_back(std::thread(gather_edges,
                std::ref(thread_edges[i]),
                std::ref(min_isocline_direction),
                std::ref(v),
                std::ref(f),
                std::ref(n),
                i * num_faces_per_thread,
                std::min(static_cast<size_t>(f.rows()), (i + 1) * num_faces_per_thread)));
    }

    for (size_t i = 0; i < processor_count; ++i) {
        threads[i].join();
    }

    for (size_t i = 0; i < processor_count; ++i) {

        size_t old_size = e0.rows();

        e0.conservativeResize(old_size + thread_edges[i].size(), e0.cols());
        e1.conservativeResize(old_size + thread_edges[i].size(), e1.cols());
        en0.conservativeResize(old_size + thread_edges[i].size(), en0.cols());
        en1.conservativeResize(old_size + thread_edges[i].size(), en1.cols());

        for (size_t j = 0; j < thread_edges[i].size(); ++j) {
            e0.row(old_size + j) = std::get<0>(thread_edges[i][j]);
            en0.row(old_size + j) = std::get<1>(thread_edges[i][j]);
            e1.row(old_size + j) = std::get<2>(thread_edges[i][j]);
            en1.row(old_size + j) = std::get<3>(thread_edges[i][j]);
        }
    }
}

float isocline_length(
        Eigen::MatrixXf& e0,
        Eigen::MatrixXf& e1) {

    assert(e0.rows() == e1.rows());

    return (e1 - e0).rowwise().norm().sum();
}

}
