#pragma once


#include "util.h"

#include "ray.h"
#include "vec3.h"

#include <cuda_runtime.h>

class camera
{
public:
  __device__ camera(int nx, int ny) :
    _x(nx),
    _y(ny)
  {
    camera::updateViewport();
  }

  __device__ int x() const { return _x; }
  __device__ int y() const { return _y; }


  __device__ ray get_ray(float u, float v) const { return ray(origin, lower_left + u * vx + v * vy - origin); }

private:
 int _x;
 int _y;
 vec3 origin = point3(0, 0, 0);
 float focal_length = 1.0;
 float viewport_height = 2.0;
 float viewport_width;
 vec3 vx;
 vec3 vy;
 vec3 lower_left;

 __device__ void updateViewport()
 {
   viewport_width = float(viewport_height * (float(_x) / float(_y)));
   vx = vec3(viewport_width, 0, 0);
   vy = vec3(0, viewport_height, 0);
   lower_left = origin - vec3(0, 0, focal_length) - vx / 2 - vy / 2;
 }
};

