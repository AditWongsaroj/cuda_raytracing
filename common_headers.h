#pragma once

// std libs
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <time.h>

// CUDA Headers
#include "cuda_fix.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// Common Headers
#include "constants.h"
#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "material.h"

#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "print_png.h"
#include "cuda_render.h"

