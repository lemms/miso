#include "isocline.h"

#include <algorithm>
#include <thread>

#include <Eigen/Dense>

#define MISO_ISOCLINE_DEBUG 0
#if MISO_ISOCLINE_DEBUG
#include <iostream>
#endif

namespace miso {

void gather_edges(
        std::vector<Eigen::Vector3f>& e0,
        std::vector<Eigen::Vector3f>& e1,
        std::vector<Eigen::Vector3f>& en0,
        std::vector<Eigen::Vector3f>& en1,
        const Eigen::Vector3f& min_isocline_direction,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n,
        const size_t start_face,
        const size_t end_face) {

    for (size_t face = start_face; face < end_face; ++face) {
        uint8_t isocline_index = 0;

        for (size_t edge = 0; edge < 3; ++edge) {
            const Eigen::Vector3f n0 = n.row(f(face, edge));
            const Eigen::Vector3f n1 = n.row(f(face, (edge + 1) % 3));

            const float a = -n0.dot(min_isocline_direction) / (n1.dot(min_isocline_direction) - n0.dot(min_isocline_direction));

            if (a >= 0.0 && a <= 1.0) {
                const Eigen::Vector3f v0 = v.row(f(face, edge));
                const Eigen::Vector3f v1 = v.row(f(face, (edge + 1) % 3));

                const Eigen::Vector3f isocline_vertex = (1.0f - a) * v0 + a * v1;
                const Eigen::Vector3f isocline_normal = (1.0f - a) * n0 + a * n1;

                // Min isocline curve passes through edge
                if (isocline_index == 0) {
                    e0.push_back(isocline_vertex);
                    en0.push_back(isocline_normal);
                } else if (isocline_index == 1) {
                    e1.push_back(isocline_vertex);
                    en1.push_back(isocline_normal);
                }

                ++isocline_index;
            }
        }

        if (isocline_index == 1) {
            // The isocline curve is degenerate on this face

#if MISO_ISOCLINE_DEBUG
            std::cerr << "Warning: Isocline curve is degenerate on face " << face << std::endl;
#endif

            e1.push_back(e0.back());
            en1.push_back(en0.back());
        }
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

    std::vector<std::vector<Eigen::Vector3f>> thread_e0(processor_count);
    std::vector<std::vector<Eigen::Vector3f>> thread_e1(processor_count);
    std::vector<std::vector<Eigen::Vector3f>> thread_en0(processor_count);
    std::vector<std::vector<Eigen::Vector3f>> thread_en1(processor_count);

    size_t num_faces_per_thread = f.rows() / processor_count;

#if MISO_ISOCLINE_DEBUG
    std::cout << "Faces per thread: " << num_faces_per_thread << std::endl;
#endif

    std::vector<std::thread> threads;

    for (size_t i = 0; i < processor_count; ++i) {
        threads.push_back(std::thread(gather_edges,
                std::ref(thread_e0[i]),
                std::ref(thread_e1[i]),
                std::ref(thread_en0[i]),
                std::ref(thread_en1[i]),
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
        assert(thread_e0[i].size() == thread_e1[i].size());
        assert(thread_e0[i].size() == thread_en0[i].size());
        assert(thread_e0[i].size() == thread_en1[i].size());

        size_t old_size = e0.rows();

        e0.conservativeResize(old_size + thread_e0[i].size(), e0.cols());
        e1.conservativeResize(old_size + thread_e1[i].size(), e1.cols());
        en0.conservativeResize(old_size + thread_en0[i].size(), en0.cols());
        en1.conservativeResize(old_size + thread_en1[i].size(), en1.cols());

        for (size_t j = 0; j < thread_e0[i].size(); ++j) {
            e0.row(old_size + j) = thread_e0[i][j];
            e1.row(old_size + j) = thread_e1[i][j];
            en0.row(old_size + j) = thread_en0[i][j];
            en1.row(old_size + j) = thread_en1[i][j];
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
