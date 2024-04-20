

#include "util.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

#include <time.h>
#include <fstream>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

const struct cam
{
  int max_x;
  int max_y;
  vec3 lower_left;
  vec3 vx;
  vec3 vy;
  vec3 origin;
};

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
  if (result)
  {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ vec3 ray_color(const ray& r, const std::unique_ptr<hittable_list>& world)
{
  hit_record rec;

    if(world->hit(r, 0, INFINITY, rec))
    {
      return 0.5 * (rec.normal + color(1, 1, 1));
    }

  vec3 unit_direction = unit_vector(r.direction());
  auto a = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, const cam &cam_view, std::unique_ptr<hittable_list>&world)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= cam_view.max_x) || (j >= cam_view.max_y)) return;
  int pixel_index = j * cam_view.max_x + i;
  float u = float(i) / float(cam_view.max_x);
  float v = float(j) / float(cam_view.max_y);
  ray r(cam_view.origin, cam_view.lower_left + u * cam_view.vx + v * cam_view.vy);
  fb[pixel_index] = ray_color(r, world);
}

__global__ void create_world(std::unique_ptr<hittable_list> &d_world)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    d_world->add(make_shared<sphere>(vec3(0, 0, -1), 0.5));
    d_world->add(make_shared<sphere>(vec3(0, -100.5, -1), 100));
  }
}

__global__ void free_world(std::unique_ptr<hittable_list> &d_world)
{
  d_world->clear();
  delete &d_world;
}

int main()
{
  //hittable** d_list;
  //checkCudaErrors(cudaMalloc((void**)&d_list, ));
  std::unique_ptr<hittable_list> d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable_list*) + 2 * sizeof(hittable*)));
  create_world KERNEL_ARGS2(1,1) (d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());





  //hittable_list world;

  //world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
  //world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));



  auto aspect_ratio = 16.0 / 9.0;
  int image_width = 1200;
  int image_height  = int(image_width / aspect_ratio);

  image_height  = (image_height  < 1) ? 1 : image_height ;

  auto focal_length = 1.0;
  auto viewport_height = 2.0;
  auto viewport_width = viewport_height * (float(image_width) / image_height );
  auto camera_center = point3(0, 0, 0);

  // Calculate the vectors across the horizontal and down the vertical viewport edges.
  auto viewport_u = vec3(viewport_width, 0, 0);
  auto viewport_v = vec3(0, viewport_height, 0);

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  // auto pixel_delta_u = viewport_u / image_width;
  // auto pixel_delta_v = viewport_v / image_height;

  // Calculate the location of the upper left pixel.
  auto viewport_upper_left = camera_center
    - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
  // auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);


  cam cam_setup{ image_width, image_height, viewport_upper_left, viewport_u, viewport_v, camera_center};

  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << image_width << "x" << image_height  << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  auto num_pixels = image_width * image_height ;
  size_t fb_size = 3 * num_pixels * sizeof(float);

  // allocate FB
  color* fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(image_width / tx + 1, image_height  / ty + 1);
  dim3 threads(tx, ty);
  render KERNEL_ARGS2(blocks, threads) (fb, cam_setup, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  std::ofstream pic;
  pic.open("out.ppm");

  char buffer[32];
  sprintf(buffer, "P3\n%d %d\n255\n", image_width, image_height );

  pic << buffer;


  for (int j = image_height  - 1; j >= 0; j--)
  {
    std::clog << "\rScan-lines remaining: " << (image_height  - j) << ' ' << std::flush;
    for (int i = 0; i < image_width; i++)
    {
      write_color(pic, fb[j*image_width + i]);
    }
  }
  pic.close();

  free_world KERNEL_ARGS2(1,1) (d_world);
  checkCudaErrors(cudaFree(fb));

}