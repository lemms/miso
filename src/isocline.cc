#include "isocline.h"

#include <algorithm>
#include <thread>
#include <tuple>

#include <Eigen/Dense>

#define MISO_ISOCLINE_DEBUG 0
#if MISO_ISOCLINE_DEBUG
#include <iostream>
#endif


namespace {
    const float threshold = 1e-7f;
}

namespace miso {

void gather_edges(
        std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>>& edges,
        const Eigen::Vector3f& isocline_direction,
        const float cos_angle,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n,
        const size_t start_face,
        const size_t end_face) {

    assert(!isnan(isocline_direction[0]));
    assert(!isnan(isocline_direction[1]));
    assert(!isnan(isocline_direction[2]));

    for (size_t face = start_face; face < end_face; ++face) {

        uint8_t isocline_index = 0;

        bool well_formed = true;
        for (size_t edge = 0; edge < 3; ++edge) {

            const uint32_t i0 = f(face, edge);

            const Eigen::Vector3f& n0 = n.row(i0);
            if (isnan(n0[0]) || isnan(n0[1]) || isnan(n0[2])) {
                well_formed = false;
            }
        }

        if (!well_formed) {
#if MISO_ISOCLINE_DEBUG
            std::cerr << "Warning: Mesh vertex normals are not well formed for face: " << face << std::endl;
#endif

            continue;
        }

        std::array<float, 3> n_dot_d {0.0f, 0.0f, 0.0f};
        std::array<float, 3> n_dot_md {0.0f, 0.0f, 0.0f};

        for (size_t edge = 0; edge < 3; ++edge) {
            const uint32_t i0 = f(face, edge);

            const Eigen::Vector3f& n0 = n.row(i0);

            n_dot_d[edge] = n0.dot(isocline_direction);
            n_dot_md[edge] = n0.dot(-isocline_direction);

            assert(!isinf(n_dot_d[edge]));
            assert(!isnan(n_dot_d[edge]));
            assert(!isinf(n_dot_md[edge]));
            assert(!isnan(n_dot_md[edge]));
        }

        for (size_t edge = 0; edge < 3; ++edge) {

            const uint32_t e0 = edge;
            const uint32_t e1 = (edge + 1) % 3;

            const uint32_t i0 = f(face, e0);
            const uint32_t i1 = f(face, e1);

            const Eigen::Vector3f& n0 = n.row(i0);
            const Eigen::Vector3f& n1 = n.row(i1);

            // Check if the edge is on the isocline curve
            if (std::abs(n_dot_d[e0] - cos_angle) < threshold && std::abs(n_dot_d[e1] - cos_angle) < threshold) {
                const Eigen::Vector3f& v0 = v.row(i0);
                const Eigen::Vector3f& v1 = v.row(i1);

                edges.emplace_back(v0, n0, v1, n1);
                isocline_index = 2;

#if MISO_ISOCLINE_DEBUG
                std::cout << "Edge is on isocline curve. Face: " << face << " edge: " << edge << std::endl;
#endif
            } else if (std::abs(n_dot_md[e0] - cos_angle) < threshold && std::abs(n_dot_md[e1] - cos_angle) < threshold) {

                const Eigen::Vector3f& v0 = v.row(i0);
                const Eigen::Vector3f& v1 = v.row(i1);

                edges.emplace_back(v0, n0, v1, n1);
                isocline_index = 2;

#if MISO_ISOCLINE_DEBUG
                std::cout << "Edge is on isocline curve. Face: " << face << " edge: " << edge << std::endl;
#endif
            }
        }

        if (isocline_index == 0) {
            for (size_t edge = 0; edge < 3; ++edge) {

                const uint32_t e0 = edge;
                const uint32_t e1 = (edge + 1) % 3;

                const uint32_t i0 = f(face, e0);
                const uint32_t i1 = f(face, e1);

                const Eigen::Vector3f& n0 = n.row(i0);
                const Eigen::Vector3f& n1 = n.row(i1);

                float a = -n_dot_d[e0] + cos_angle;
                if (std::abs(n_dot_d[e1] - n_dot_d[e0]) > threshold) {
                    a /= n_dot_d[e1] - n_dot_d[e0];
                }

                float ma = -n_dot_md[e0] + cos_angle;
                if (std::abs(n_dot_md[e1] - n_dot_md[e0]) > threshold) {
                    ma /= n_dot_md[e1] - n_dot_md[e0];
                }

                assert(!isnan(n0[0]));
                assert(!isnan(n0[1]));
                assert(!isnan(n0[2]));
                assert(!isnan(n1[0]));
                assert(!isnan(n1[1]));
                assert(!isnan(n1[2]));

                assert(!isnan(a));
                assert(!isnan(ma));
                assert(!isinf(a));
                assert(!isinf(ma));

                if (a >= 0.0 && a <= 1.0) {
                    const Eigen::Vector3f& v0 = v.row(i0);
                    const Eigen::Vector3f& v1 = v.row(i1);

                    const Eigen::Vector3f isocline_vertex = (1.0f - a) * v0 + a * v1;
                    const Eigen::Vector3f isocline_normal = (1.0f - a) * n0 + a * n1;

                    assert(!isnan(isocline_vertex[0]));
                    assert(!isnan(isocline_vertex[1]));
                    assert(!isnan(isocline_vertex[2]));
                    assert(!isnan(isocline_normal[0]));
                    assert(!isnan(isocline_normal[1]));
                    assert(!isnan(isocline_normal[2]));

                    // Min isocline curve passes through edge
                    if (isocline_index == 0) {
                        edges.emplace_back(isocline_vertex, isocline_normal, isocline_vertex, isocline_normal);
                        ++isocline_index;
                    } else if (isocline_index == 1) {
                        std::get<2>(edges[edges.size() - 1]) = isocline_vertex;
                        std::get<3>(edges[edges.size() - 1]) = isocline_normal;

                        ++isocline_index;

                        break;
                    }
                } else if (ma >= 0.0 && ma <= 1.0) {
                    const Eigen::Vector3f& v0 = v.row(i0);
                    const Eigen::Vector3f& v1 = v.row(i1);

                    const Eigen::Vector3f isocline_vertex = (1.0f - ma) * v0 + ma * v1;
                    const Eigen::Vector3f isocline_normal = (1.0f - ma) * n0 + ma * n1;

                    assert(!isnan(isocline_vertex[0]));
                    assert(!isnan(isocline_vertex[1]));
                    assert(!isnan(isocline_vertex[2]));
                    assert(!isnan(isocline_normal[0]));
                    assert(!isnan(isocline_normal[1]));
                    assert(!isnan(isocline_normal[2]));

                    // Min isocline curve passes through edge
                    if (isocline_index == 0) {
                        edges.emplace_back(isocline_vertex, isocline_normal, isocline_vertex, isocline_normal);
                        ++isocline_index;
                    } else if (isocline_index == 1) {
                        std::get<2>(edges[edges.size() - 1]) = isocline_vertex;
                        std::get<3>(edges[edges.size() - 1]) = isocline_normal;

                        ++isocline_index;

                        break;
                    }
                }
            }
        }

        if (isocline_index == 1) {
            // The isocline curve is degenerate on this face

#if MISO_ISOCLINE_DEBUG
            std::cerr << "Warning: Isocline curve is degenerate on face " << face << std::endl;
#endif
            edges.pop_back();
        }
    }
}

void compute_isocline(
        Eigen::MatrixXf& e0,
        Eigen::MatrixXf& e1,
        Eigen::MatrixXf& en0,
        Eigen::MatrixXf& en1,
        const Eigen::Vector3f& isocline_direction,
        const float cos_angle,
        const Eigen::MatrixXf& v,
        const Eigen::MatrixXi& f,
        const Eigen::MatrixXf& n) {

    const auto processor_count = std::thread::hardware_concurrency();

#if MISO_ISOCLINE_DEBUG
    std::cout << "Processor count: " << processor_count << std::endl;
#endif

    std::vector<std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>>> thread_edges(processor_count);

    size_t num_faces_per_thread = (f.rows() + (processor_count - 1)) / processor_count;

#if MISO_ISOCLINE_DEBUG
    std::cout << "Faces per thread: " << num_faces_per_thread << std::endl;
#endif

    std::vector<std::thread> threads;

    for (size_t i = 0; i < processor_count; ++i) {
        threads.push_back(std::thread(gather_edges,
                std::ref(thread_edges[i]),
                std::ref(isocline_direction),
                cos_angle,
                std::ref(v),
                std::ref(f),
                std::ref(n),
                i * num_faces_per_thread,
                std::min(static_cast<size_t>(f.rows()), (i + 1) * num_faces_per_thread)));

#if MISO_ISOCLINE_DEBUG
        if (i == processor_count - 1) {
            std::cout << "Last thread range:" << std::endl;
            std::cout << i * num_faces_per_thread << " to " << std::min(static_cast<size_t>(f.rows()), (i + 1) * num_faces_per_thread) << std::endl;
            std::cout << "Num faces: " << f.rows() << std::endl;
        }
#endif
    }

    for (size_t i = 0; i < processor_count; ++i) {
        threads[i].join();
    }

    for (size_t i = 0; i < processor_count; ++i) {

        size_t old_size = e0.rows();

        e0.conservativeResize(old_size + thread_edges[i].size(), e0.cols());
        en0.conservativeResize(old_size + thread_edges[i].size(), en0.cols());
        e1.conservativeResize(old_size + thread_edges[i].size(), e1.cols());
        en1.conservativeResize(old_size + thread_edges[i].size(), en1.cols());

        for (size_t j = 0; j < thread_edges[i].size(); ++j) {
            e0.row(old_size + j) = std::get<0>(thread_edges[i][j]);
            en0.row(old_size + j) = std::get<1>(thread_edges[i][j]);
            e1.row(old_size + j) = std::get<2>(thread_edges[i][j]);
            en1.row(old_size + j) = std::get<3>(thread_edges[i][j]);
        }
    }

#if MISO_ISOCLINE_DEBUG
    std::cout << "e0: " << e0.rows() << ", " << e0.cols() << std::endl;
    std::cout << "e1: " << e1.rows() << ", " << e1.cols() << std::endl;
    std::cout << "en0: " << en0.rows() << ", " << en0.cols() << std::endl;
    std::cout << "en1: " << en1.rows() << ", " << en1.cols() << std::endl;
#endif
}

float isocline_length(
        Eigen::MatrixXf& e0,
        Eigen::MatrixXf& e1) {

    assert(e0.rows() == e1.rows());

    return (e1 - e0).rowwise().norm().sum();
}

}
