#pragma once

// Defines
#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

#include "common_headers.h"
#include "hittable.h"


__device__ vec3 random_unit_vector(curandState* local_rand_state)
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
    const override
  {
    auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

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
  __device__ explicit metal(const vec3& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1){}

  __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state)
    const override
  {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_rand_state));
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

private:
  vec3 albedo;
  float fuzz;
};

class dielectric : public material {
  public:
    __device__ explicit dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state)
    const final{
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        float ri = rec.front_face ? (1.0f/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = std::min(dot(-unit_direction, rec.normal), 1.0f);
        auto sin_theta = float(sqrt(1.0 - cos_theta * cos_theta));

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(local_rand_state))
          direction = reflect(unit_direction, rec.normal);
        else
          direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);

        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __device__ static float reflectance(float cosine, float refraction_index)
    {
      // Use Schlick's approximation for reflectance.
      auto r0 = (1 - refraction_index) / (1 + refraction_index);
      r0 = r0 * r0;
      return r0 + (1 - r0) * float(pow((1 - cosine), 5));
    }
};