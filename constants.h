#pragma once
#include "vec3.h"

// Constant
const int samples_per_pixel = 64;
const int tx = 16;
const int ty = 16;

const int sphere_total = 1 + 22 * 22 + 3;

constexpr int image_width = 1200;
constexpr float aspect_ratio = 16.0f / 9.0f;
constexpr int image_height = int(image_width / aspect_ratio);

constexpr int num_pixels = image_width * image_height;
constexpr size_t fb_size = num_pixels * sizeof(vec3);


const float pi = std::numbers::pi_v<float>;


// Utility Functions
__device__ inline float degrees_to_radians(float degrees)
{
  return degrees * std::numbers::pi_v<float> / 180.0f;
}