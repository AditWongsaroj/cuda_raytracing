#pragma once


#include "util.h"

#include "ray.h"
#include "vec3.h"

#include <cuda_runtime.h>

class camera
{
public:
  __device__ camera(int _x, int _y) :
    x(_x),
    y(_y),
    viewport_width(float(viewport_height* (float(x) / float(y)))),
    vx(vec3(viewport_width, 0, 0)),
    vy(vec3(0, viewport_height, 0)),
    lower_left(origin - vec3(0, 0, focal_length) - vx / 2 - vy / 2)
  {
  }

  __device__ ray get_ray(float u, float v) const { return ray(origin, lower_left + u * vx + v * vy - origin); }


  int x;
  int y;
  vec3 origin = point3(0, 0, 0);




private:

  /* Private Camera Variables Here */
 float focal_length = 1.0;
 float viewport_height = 2.0;
 float viewport_width;
 vec3 vx;
  vec3 vy;

public:

  vec3 lower_left;
};

