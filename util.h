#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <fstream>

// Common Headers


#include "camera.h"
#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"


// Cuda Headers

#include "cuda_fix.h" // replaces <<< >>> with KERNAL#ofArgs
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const float pi = 3.1415926535897932385f;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}
