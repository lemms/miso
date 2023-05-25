cmake_minimum_required(VERSION 3.7...3.24)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
#igl_include(glfw)

project(ISOCLINE VERSION 1.0
    DESCRIPTION "Minimum Isocline"
    LANGUAGES CXX)

add_executable(min_isocline
    src/isocline.cc
    src/isocline.h
    src/main.cc
    src/solver.cc
    src/solver.h
)

target_compile_features(min_isocline PRIVATE cxx_std_17)

target_link_libraries(min_isocline
    Eigen3::Eigen
    igl::core
)
