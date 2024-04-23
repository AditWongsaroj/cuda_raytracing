#pragma once


#include "util.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
  if (result)
  {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}



__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int x, int y)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    *d_list = new sphere(vec3(0, 0, -1), 0.5f);
    *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
    *d_world = new hittable_list(d_list, 2);
    *d_camera = new camera(x,y);

  }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera** d_camera)
{
  delete* d_list;
  delete *(d_list + 1);
  delete *d_world;
  delete *d_camera;
}

__device__ color ray_color(const ray &r, hittable **world)
{
  hit_record rec;
  if ((*world)->hit(r, interval(0.0, FLT_MAX), rec))
  {
    return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
  }
  else
  {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
  }
}


__global__ void render(int ns, vec3* fb, camera **cam_view, hittable **world, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= (*cam_view)->x()) || (j >= (*cam_view)->y())) return;
  int pixel_index = j * (*cam_view)->x() + i;

  curand_init(2024, pixel_index, 0, &rand_state[pixel_index]);
  curandState local_rand_state = rand_state[pixel_index];

  color col(0,0,0);
  for (int s = 0; s < ns; s++)
  {
    float u = (float(i) + curand_uniform(&local_rand_state)) / float((*cam_view)->x());
    float v =( float(j) + curand_uniform(&local_rand_state)) / float((*cam_view)->y());
    ray r = (*cam_view)->get_ray(u, v);
    col = col + ray_color(r, world);
  }
  fb[pixel_index] = col / float(ns);
}
