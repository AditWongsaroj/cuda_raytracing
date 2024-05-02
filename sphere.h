#pragma once
#include "hittable.h"
#include "material.h"

class sphere : public hittable
{
public:
  __device__ sphere() {}
  __device__ sphere(vec3 cen, float r, material* mat) : center(cen), radius(r), mat_ptr(mat) {};

  __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;
  vec3 center;
  float radius{};
  material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const
{
  vec3 oc = center - r.origin();
  auto a = r.direction().squared_length();
  auto h = dot(r.direction(), oc);
  auto c = oc.squared_length() - radius * radius;

  auto discriminant = h * h - a * c;
  if (discriminant < 0)
    return false;

  auto sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (h - sqrtd) / a;
  if (!ray_t.surrounds(root))
  {
    root = (h + sqrtd) / a;
    if (!ray_t.surrounds(root))
      return false;
  }

  rec.t = root;
  rec.p = r.at(rec.t);
  vec3 outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mat_ptr;

  return true;
}