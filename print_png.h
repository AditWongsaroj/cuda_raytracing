#pragma once

#include <fstream>
const int N = 1 << 20;

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

      auto r = int(fb[pixel_index].r());
      auto g = int(fb[pixel_index].g());
      auto b = int(fb[pixel_index].b());

      pic << r << " " << g << " " << b << "\n";
    }
  }
  pic.close();
}