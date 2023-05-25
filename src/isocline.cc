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
        std::vector<Eigen::Vector3d>& e0,
        std::vector<Eigen::Vector3d>& e1,
        std::vector<Eigen::Vector3d>& en0,
        std::vector<Eigen::Vector3d>& en1,
        const Eigen::Vector3d& min_isocline_direction,
        const Eigen::MatrixXd& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXd& n,
        const size_t start_face,
        const size_t end_face) {

    for (size_t face = start_face; face < end_face; ++face) {
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
                    e0.push_back(isocline_vertex);
                    en0.push_back(isocline_normal);
                } else {
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
        Eigen::MatrixXd& e0,
        Eigen::MatrixXd& e1,
        Eigen::MatrixXd& en0,
        Eigen::MatrixXd& en1,
        const Eigen::Vector3d& min_isocline_direction,
        const Eigen::MatrixXd& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXd& n) {

    const auto processor_count = std::thread::hardware_concurrency();

#if MISO_ISOCLINE_DEBUG
    std::cout << "Processor count: " << processor_count << std::endl;
#endif

    std::vector<std::vector<Eigen::Vector3d>> thread_e0(processor_count);
    std::vector<std::vector<Eigen::Vector3d>> thread_e1(processor_count);
    std::vector<std::vector<Eigen::Vector3d>> thread_en0(processor_count);
    std::vector<std::vector<Eigen::Vector3d>> thread_en1(processor_count);

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

double isocline_length(
        Eigen::MatrixXd& e0,
        Eigen::MatrixXd& e1) {

    assert(e0.rows() == e1.rows());

    return (e1 - e0).rowwise().norm().sum();
}

}
