#pragma once

#include "common_headers.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
      static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();

    exit(99);
  }
}


__device__ vec3 color(const ray& r, hittable** world, curandState* local_rand_state)
{
  const int max_iterations = 8;

  ray cur_ray = r;
  auto cur_attenuation = vec3(1.0, 1.0, 1.0);

  for (int i = 0; i < max_iterations; i++)
  {
    hit_record rec;
    if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec))
    {
      ray scattered;
      vec3 attenuation;

      if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
      {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else
      {
        return vec3(0.0f, 0.0f, 0.0f);
      }
    }
    else
    {
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
      return cur_attenuation * c;
    }
  }
  return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    curand_init(1984, 0, 0, rand_state);
  }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x + i;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(2024, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int3 size_init, int ns,
    camera** cam, hittable** world, curandState* rand_state)
{
  int min_x = size_init.x;
  int max_x = size_init.y;
  int max_y = size_init.z;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i < min_x) || (i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++)
  {
    float u = float(float(i) + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(float(j)+ curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col += color(r, world, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;


  const interval intensity(0.000f, 0.999f);

  col /= float(ns);
  col[0] = 256 * intensity.clamp(sqrt(col[0]));
  col[1] = 256 * intensity.clamp(sqrt(col[1]));
  col[2] = 256 * intensity.clamp(sqrt(col[2]));
  fb[pixel_index] = col;
}

