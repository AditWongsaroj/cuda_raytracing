

#include "util.h"

#include "cuda_functions.h"

#include <time.h>

int main()
{
  float aspect_ratio = 16.0f / 9.0f;

  int image_width = 1200;
  int image_height = int(image_width / aspect_ratio);
  image_height = image_height < 1 ? 1 : image_height;

  int samples = 100;
  int tx = 8;
  int ty = 8;

  auto num_pixels = image_width * image_height ;
  size_t fb_size = 3ll * num_pixels * sizeof(float);

  dim3 blocks(image_width / tx + 1, image_height  / ty + 1);
  dim3 threads(tx, ty);

  std::cerr << "Rendering a " << image_width << "x" << image_height  << " image ";
  std::cerr << "with " << samples << " samples per pixel ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  // allocate FrameBuffer
  color* fb{};
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

  // World
  hittable** d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, 2* sizeof(hittable*)));
  hittable** d_world{};
  checkCudaErrors(cudaMalloc((void**)&d_world, 1*  sizeof(hittable*)));
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

  create_world KERNEL_ARGS2(1,1)(d_list, d_world, d_camera, image_width, image_height);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Render our buffer

  render KERNEL_ARGS2(blocks, threads) (samples, fb, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render_out(fb, image_width, image_height);

  //Clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world KERNEL_ARGS2(1,1) (d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(fb));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}