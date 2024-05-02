#pragma once

#include <fstream>
#include "vec3.h"

inline float linear_to_gamma(float linear_component)
{
  if (linear_component > 0)
    return sqrt(linear_component);

  return 0;
}

void print_pgn(const vec3* fb, int nx, int ny)
{
  std::ofstream pic;
  pic.open("out.ppm");
  pic << "P3\n" << nx << ' ' << ny << "\n255\n";

  for (int j = ny - 1; j >= 0; j--)
  {
    for (int i = 0; i < nx; i++)
    {
      size_t pixel_index = j * nx + i;

      float r = fb[pixel_index].r();
      float g = fb[pixel_index].g();
      float b = fb[pixel_index].b();

      // Apply a linear to gamma transform for gamma 2
      r = linear_to_gamma(r);
      g = linear_to_gamma(g);
      b = linear_to_gamma(b);


      static const interval intensity(0.000f, 0.999f);

      int ir = int(256 * intensity.clamp(r));
      int ig = int(256 * intensity.clamp(g));
      int ib = int(256 * intensity.clamp(b));
      pic << ir << " " << ig << " " << ib << "\n";
    }
  }
  pic.close();
}