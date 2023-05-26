# miso
Miso: The Minimum Isocline Curve Solver

Miso is a solver which will output the direction vector which produces the minimum length [isocline curve](http://www2.me.rochester.edu/courses/ME204/nx_help/index.html#uid:points_curves_crv_isocline) for a given mesh.

Miso uses (simulated annealing)[https://en.wikipedia.org/wiki/Simulated_annealing] on randomly generated candidate neighbor directions with a [multiplicative cooling schedule](https://en.wikipedia.org/wiki/Simulated_annealing#Cooling_schedule) to find an approximate global minimum isocline curve direction.

Note: Miso currently only supports 0-degree isocline curves.

# Pre-requisites

Eigen3
libigl

# Building from source

1. In the miso directory, create a build directory:

```
mkdir build
cd build
```

2. Run cmake

```
cmake -DCMAKE_BUILD_TYPE=Release ..
```

3. Build Miso

```
cmake --build .
```

4. Install Miso

```
cmake --install .
```

# Usage

miso <mesh file> --min-temp=<min temperature> --alpha=<alpha> --max-iter=<max iterations> --max-inner-iter=<max inner iterations> --neighbor-stddev=<neighbor standard deviation> --verbose --debug-files

## Arguments

Mesh file:

The input mesh file. Supported file formats: OBJ, OFF, PLY, STL.

--min-temp:

The minimum temperature termination condition for the simulated annealing cooling schedule.

--alpha:

The cooling rate for the simulated annealing cooling schedule.

--max-iter:

The maximum iterations to run for simulated annealing.

--max-inner-iter:

The maximum number of iterations to run per cooling cycle.

--neighbor-stddev:

The standard deviation of the gaussian distribution used to find neighbor samples.

--verbose:

Output verbose logs messages.

--debug-files:

Dump debug files to a temp directory.
