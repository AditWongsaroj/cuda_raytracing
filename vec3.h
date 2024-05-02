#pragma once
#include "common_headers.h"

class vec3
{
public:
  __host__ __device__ vec3() {}
  __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
  __host__ __device__ inline float x() const { return e[0]; }
  __host__ __device__ inline float y() const { return e[1]; }
  __host__ __device__ inline float z() const { return e[2]; }
  __host__ __device__ inline float r() const { return e[0]; }
  __host__ __device__ inline float g() const { return e[1]; }
  __host__ __device__ inline float b() const { return e[2]; }

  __host__ __device__ inline const vec3& operator+() const { return *this; }
  __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }
  __host__ __device__ inline float& operator[](int i) { return e[i]; };

  __host__ __device__ inline vec3& operator+=(const vec3& v2);
  __host__ __device__ inline vec3& operator-=(const vec3& v2);
  __host__ __device__ inline vec3& operator*=(const vec3& v2);
  __host__ __device__ inline vec3& operator/=(const vec3& v2);
  __host__ __device__ inline vec3& operator*=(const float t);
  __host__ __device__ inline vec3& operator/=(const float t);

  __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
  __host__ __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
  __host__ __device__ inline void make_unit_vector();

  __host__ __device__ inline bool near_zero() const
  {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
  }

  float e[3]{};
};



inline std::istream& operator>>(std::istream& is, vec3& t)
{
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t)
{
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
  float k = 1.0f / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v)
{
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
  return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
  return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
  return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
    (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
    (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3& v)
{
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v)
{
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v)
{
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
  float k = 1.0f / t;

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

__device__ inline vec3 unit_vector(vec3 v)
{
  return v / v.length();
}

__device__ inline vec3 random_in_unit_disk(curandState* local_rand_state)
{
  while (true)
  {
    auto p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    if (p.squared_length() < 1)
      return p;
  }
}


__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
  return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = std::min(dot(-uv, n), 1.0f);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.squared_length())) * n;
    return r_out_perp + r_out_parallel;
}