#include "common_headers.h"
#include "cuda_render.h"
#include "scene.h"

static void RunRayTracer(){

  std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel << " samples per pixel ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  // allocate FB
  vec3* fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  // allocate random state
  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
  curandState* d_rand_state_for_material;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state_for_material, 1 * sizeof(curandState)));

  rand_init KERNEL_ARGS2(1, 1) (d_rand_state_for_material);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make our world of hittable objs & the camera
  hittable** d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, sphere_total * sizeof(hittable*)));
  hittable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
  create_world KERNEL_ARGS2(1, 1) (d_list, d_world, d_camera, sphere_total, d_rand_state_for_material);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(image_width / tx + 1, image_height / ty + 1);
  dim3 threads(tx, ty);
  render_init KERNEL_ARGS2(blocks, threads) (image_width, image_height, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  //int streams = 2;
  //int stream_width = image_width / 2;

  int3 size_init = { 0 , image_width, image_height };


  render KERNEL_ARGS2(blocks, threads) (fb, size_init, samples_per_pixel, d_camera, d_world, d_rand_state);
  // render KERNEL_ARGS2( blocks, threads ) (fb, 0, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaDeviceSynchronize());
   stop = clock();
   double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
   std::cerr << "took " << timer_seconds << " seconds.\n";

   // Output FB as Image

   print_pgn(fb, image_width, image_height);
   // clean up
   checkCudaErrors(cudaDeviceSynchronize());
   free_world KERNEL_ARGS2(1, 1) (d_list, d_world, d_camera, sphere_total);
   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaFree(d_camera));
   checkCudaErrors(cudaFree(d_world));
   checkCudaErrors(cudaFree(d_list));
   checkCudaErrors(cudaFree(d_rand_state));
   checkCudaErrors(cudaFree(fb));

   cudaDeviceReset();
}