#pragma once

// Defines
#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

// Constant
constexpr int image_width = 1200;
constexpr float aspect_ratio = 16.0f/9.0f;

constexpr int image_height = int(image_width / aspect_ratio);

const float pi = std::numbers::pi_v<float>;


// Utility Functions
__device__ inline float degrees_to_radians(float degrees)
{
  return degrees * std::numbers::pi_v<float> / 180.0f;
}
