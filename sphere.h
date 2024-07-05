#pragma once

class sphere : public hittable
{
public:
  __device__ sphere() {};
  __device__ sphere(vec3 cen, float r, material* mat) : center(cen), radius(r), mat_ptr(mat) {};

  __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const {
    vec3 oc = center - r.origin();
    float a = r.direction().squared_length();
    float h = dot(r.direction(), oc);
    float c = oc.squared_length() - radius * radius;

    float discriminant = h * h - a * c;
    if (discriminant < 0)
      return false;

    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (h - sqrtd) / a;
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
  };


  material* mat_ptr{};
private:
  vec3 center;
  float radius{};
};

