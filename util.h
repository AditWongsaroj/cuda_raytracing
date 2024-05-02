#pragma once
// Cuda Headers
#include <cmath>
#include <fstream>
#include <memory>
#include <numbers>
#include <time.h>

#include "cuda_fix.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// Constant
const float pi = std::numbers::pi_v<float>;
const float infinity = INFINITY;

// C++ Std Usings
using std::sqrt;

// Utility Functions
inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

// Common Headers
#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "material.h"

#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "print_pgn.h"
#include "cuda_functions.h"


