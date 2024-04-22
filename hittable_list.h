#pragma once

#include "util.h"
#include "hittable.h"

#include <vector>

class hittable_list : public hittable
{
public:
  hittable** objects;
  int list_size;

  __device__ hittable_list() {}
  __device__ hittable_list(hittable** o, int n) : objects(o), list_size(n) {};


  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override
  {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (int i = 0; i < list_size; i++)
    {
      if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
      {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};
