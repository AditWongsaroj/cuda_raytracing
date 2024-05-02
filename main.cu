#include "util.h"

int main()
{
  int nx = 1200;
  int ny = 600;
  int ns = 100;
  int tx = 16;
  int ty = 16;

  std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3* fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  // allocate random state
  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

  // make our world of hittable objs & the camera
  hittable** d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
  hittable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
  create_world KERNEL_ARGS2( 1, 1 ) (d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render_init KERNEL_ARGS2( blocks, threads ) (nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render KERNEL_ARGS2(blocks, threads ) (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image

  print_pgn(fb, nx, ny);
  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world KERNEL_ARGS2( 1, 1 ) (d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();
}
