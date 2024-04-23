#pragma once

#include "util.h"
#include "hittable.h"

class sphere : public hittable
{
public:
  __device__ sphere(const point3& center, double radius) : center(center), radius(static_cast<float>(radius)) {}
  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

private:
  point3 center;
  float radius;
};

__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const
{
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;

    if (auto discriminant = h * h - a * c; discriminant > 0)
    {
      auto temp = float((h - sqrt(discriminant)) / a);
      if (temp < ray_t.max && temp > ray_t.min)
      {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        return true;
      }
      temp = float((h - sqrt(discriminant)) / a);
      if (temp < ray_t.max && temp > ray_t.min)
      {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        return true;
      }
    }
    return false;
}