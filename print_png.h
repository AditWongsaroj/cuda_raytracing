#pragma once

#include <fstream>
#include "vec3.h"

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

      int r = int(fb[pixel_index].r());
      int g = int(fb[pixel_index].g());
      int b = int(fb[pixel_index].b());

      pic << r << " " << g << " " << b << "\n";
    }
  }
  pic.close();
}