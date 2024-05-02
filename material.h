#pragma once

#include "util.h"
#include "hittable.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))


__device__ vec3 random_in_unit_sphere(curandState* local_rand_state)
{
  vec3 p;
  do
  {
    p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
  } while (p.squared_length() >= 1.0f);
  return p;
}

class material
{
public:
  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const
  {
    return false;
  }
};

class lambertian : public material
{
public:
  __device__ explicit lambertian(const vec3& albedo) : albedo(albedo) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state)
    const
  {
    auto scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);

    if (scatter_direction.near_zero())
      scatter_direction = rec.normal;


    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }

private:
  vec3 albedo;
};


class metal : public material
{
public:
  __device__ explicit metal(const vec3& albedo) : albedo(albedo) {}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,curandState* local_rand_state)
    const
  {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return true;
  }

private:
  vec3 albedo;
};