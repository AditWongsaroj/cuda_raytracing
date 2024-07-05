#pragma once
#include "common_headers.h"

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable** d_list, hittable** d_world,
  camera** d_camera, int sphere_count, curandState* rand_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    curandState local_rand_state = *rand_state;


    material* m_ground = new lambertian(vec3(0.5f, 0.5f, 0.5f));
    d_list[0] = new sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, m_ground);

    int i = 1;
    for (int j = -11; j < 11; j++)
    {
      for (int k = -11; k < 11; k++)
      {
        float rand_mat = RND;
        vec3 center(float(j) + RND, 0.2f, float(k) + RND);
        if (rand_mat < 0.8f)
        {
          material* _lamb = new lambertian(vec3(RND * RND, RND * RND, RND * RND));
          d_list[i++] = new sphere(center, 0.2f, _lamb);
        }
        else if (rand_mat < 0.95f)
        {
          material* _metal = new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND);
          d_list[i++] = new sphere(center, 0.2f, _metal);
        }
        else
        {
          material* _dielec = new dielectric(1.5);
          d_list[i++] = new sphere(center, 0.2f, _dielec);
        }

      }
    }

    material* m_lamb = new lambertian(vec3(0.4f, 0.2f, 0.1f));
    material* m_diel = new dielectric(1.5f);
    material* m_metal = new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f);

    d_list[i++] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, m_diel);
    d_list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, m_lamb);
    d_list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, m_metal);
    *rand_state = local_rand_state;
    *d_world = new hittable_list(d_list, sphere_count);

    //camera settings
    auto vfov = 30.0f;
    auto lookfrom = vec3(13, 2, 3);
    auto lookat = vec3(0, 0, 0);
    auto vup = vec3(0, 3, 0);

    float focus_dist = (lookfrom - lookat).length();
    float aperture = 0.2f;

    *d_camera = new camera(lookfrom,
      lookat,
      vup,
      vfov, focus_dist, aperture);
  }
}

__global__ void free_world(hittable** d_list, hittable** d_world,
  camera** d_camera, int sphere_count)
{
  for (int i = 0; i < sphere_count; i++)
  {
    delete ((sphere*)d_list[i])->mat_ptr;
    delete* (d_list + i);
  }

  delete* d_world;
  delete* d_camera;
}
