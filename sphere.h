#pragma once

#include "util.h"
#include "hittable.h"

class sphere : public hittable
{
public:

  __device__ sphere() {}
  __device__ sphere(const point3& center, double radius) : center(center), radius(radius) {}

  __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;


private:
  point3 center;
  double radius;
};

__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const
{
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = h * h - a * c;

    if (discriminant > 0)
    {
      float temp = (h - sqrt(discriminant)) / a;
      if (temp < ray_t.max && temp > ray_t.min)
      {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        return true;
      }
      temp = (h + sqrt(discriminant)) / a;
      if (temp < ray_t.max && temp > ray_t.min)
      {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        return true;
      }
    }
    return false;


    // if (discriminant < 0)
    //   return false;
    //
    // auto sqrtd = sqrt(discriminant);
    //
    // // Find the nearest root that lies in the acceptable range.
    // auto root = (h - sqrtd) / a;
    // if (!ray_t.surrounds(root))
    // {
    //   root = (h + sqrtd) / a;
    //   if (!ray_t.surrounds(root))
    //     return false;
    // }
    //
    // rec.t = root;
    // rec.p = r.at(rec.t);
    // vec3 outward_normal = (rec.p - center) / radius;
    // rec.set_face_normal(r, outward_normal);
    //
    // return true;
}