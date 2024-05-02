#include "common_headers.h"

#define RND (curand_uniform(&local_rand_state))
__global__ void create_world(hittable** d_list, hittable** d_world,
      camera** d_camera, int sphere_count, curandState* rand_state)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    curandState local_rand_state = *rand_state;


    material* m_ground = new lambertian(vec3(0.5f, 0.5f, 0.5f));
    d_list[0] = new sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, m_ground);

    int i = 1;
    for (int j = -11; j < 11; j++)
    {
      for (int k = -11; k < 11; k++)
      {
        float rand_mat = RND;
        vec3 center(j + RND, 0.2f, k + RND);
        if (rand_mat < 0.8f)
        {
          material* _lamb = new lambertian(vec3(RND * RND, RND * RND, RND * RND));
          d_list[i++] = new sphere(center, 0.2, _lamb);
        }
        else if( rand_mat < 0.95f)
        {
          material* _metal = new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND);
          d_list[i++] = new sphere(center, 0.2, _metal);
        }
        else
        {
          d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
        }

      }
    }



    material* m_lamb = new lambertian(vec3(0.4f, 0.2f, 0.1f));
    material* m_diel = new dielectric(1.5f);
    material* m_metal = new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f);

    d_list[i++] = new sphere(vec3( 0.0f, 1.0f, 0.0f), 1.0f, m_diel);
    d_list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, m_lamb);
    d_list[i++] = new sphere(vec3( 4.0f, 1.0f, 0.0f), 1.0f, m_metal);
    *rand_state = local_rand_state;
    *d_world = new hittable_list(d_list, sphere_count);

    //camera settings
    auto vfov     = 30.0f;
    auto lookfrom = vec3(13, 2, 3);
    auto lookat   = vec3(0,0,0);
    auto vup      = vec3(0,3,0);

    float focus_dist = (lookfrom - lookat).length();
    float aperture = 0.2f;

    *d_camera = new camera(lookfrom,
      lookat,
      vup,
      vfov, focus_dist, aperture);
  }
}

__global__ void free_world(hittable** d_list, hittable** d_world,
                            camera** d_camera, int sphere_count)
{
  for (int i = 0; i < sphere_count; i++)
  {
    delete ((sphere*)d_list[i])->mat_ptr;
    delete* (d_list + i);
  }

  delete* d_world;
  delete* d_camera;
}


int main()
{
  const int samples_per_pixel = 100;
  const int tx = 16;
  const int ty = 16;

  const int sphere_count = 1 + 22*22 + 3;

  std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel << " samples per pixel ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = image_width * image_height;
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3* fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  // allocate random state
  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
  curandState *d_rand_state_for_material;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state_for_material, 1*sizeof(curandState)));

  rand_init KERNEL_ARGS2(1, 1) (d_rand_state_for_material);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make our world of hittable objs & the camera
  hittable** d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, sphere_count * sizeof(hittable*)));
  hittable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
  create_world KERNEL_ARGS2( 1, 1 ) (d_list, d_world, d_camera, sphere_count, d_rand_state_for_material);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(image_width / tx + 1, image_height / ty + 1);
  dim3 threads(tx, ty);
  render_init KERNEL_ARGS2( blocks, threads ) (image_width, image_height, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render KERNEL_ARGS2( blocks, threads ) (fb, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image

  print_pgn(fb, image_width, image_height);
  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world KERNEL_ARGS2( 1, 1 ) (d_list, d_world, d_camera, sphere_count);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();
}
