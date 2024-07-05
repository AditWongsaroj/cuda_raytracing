#pragma once

class vec3
{
public:
  __host__ __device__ vec3(): e(make_float3(0,0,0)) {};
  __host__ __device__ vec3(float e0, float e1, float e2): e(make_float3(e0,e1,e2)) {};
  __host__ __device__ inline float x() const { return e.x; };
  __host__ __device__ inline float y() const { return e.y; };
  __host__ __device__ inline float z() const { return e.z; };
  __host__ __device__ inline float r() const { return e.x; };
  __host__ __device__ inline float g() const { return e.y; };
  __host__ __device__ inline float b() const { return e.z; };

  __host__ __device__ inline const vec3& operator+() const { return *this; };
  __host__ __device__ inline vec3 operator-() const { return vec3(-e.x, -e.y, -e.z); };
  __host__ __device__ inline float operator[](int i) const {
    switch (i)
    {
    case 0:
      return e.x;
      break;
    case 1:
      return e.y;
      break;
    case 2:
      return e.z;
      break;
    default: __assume(false);
    }
    return e.x; // unreached, to stop warnings
  };
  __host__ __device__ inline float& operator[](int i) {
    switch (i)
    {
    case 0:
      return e.x;
      break;
    case 1:
      return e.y;
      break;
    case 2:
      return e.z;
      break;
    default: __assume(false);
    }
    return e.x; // unreached, to stop warnings
  };

  __host__ __device__ vec3& operator+=(const vec3& v2);
  __host__ __device__ vec3& operator-=(const vec3& v2);
  __host__ __device__ vec3& operator*=(const vec3& v2);
  __host__ __device__ vec3& operator/=(const vec3& v2);
  __host__ __device__ vec3& operator*=(const float t);
  __host__ __device__ vec3& operator/=(const float t);

  __host__ __device__ inline float length() const { return sqrt(e.x * e.x + e.y * e.y + e.z * e.z); }
  __host__ __device__ inline float squared_length() const { return e.x * e.x + e.y * e.y + e.z * e.z; }

  __host__ __device__ inline bool near_zero() const
  {
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (std::fabs(e.x) < s) && (std::fabs(e.y) < s) && (std::fabs(e.z) < s);
  }

  float3 e;
};



inline std::istream& operator>>(std::istream& is, vec3& t)
{
  is >> t.e.x >> t.e.y >> t.e.z;
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t)
{
  os << t.e.x << " " << t.e.y << " " << t.e.z;
  return os;
}


__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e.x + v2.e.x, v1.e.y + v2.e.y, v1.e.z + v2.e.z);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e.x - v2.e.x, v1.e.y - v2.e.y, v1.e.z - v2.e.z);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e.x * v2.e.x, v1.e.y * v2.e.y, v1.e.z * v2.e.z);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
  return vec3(v1.e.x / v2.e.x, v1.e.y / v2.e.y, v1.e.z / v2.e.z);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v)
{
  return vec3(t * v.e.x, t * v.e.y, t * v.e.z);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
  return vec3(v.e.x / t, v.e.y / t, v.e.z / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
  return vec3(t * v.e.x, t * v.e.y, t * v.e.z);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
  return v1.e.x * v2.e.x + v1.e.y * v2.e.y + v1.e.z * v2.e.z;
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
  return vec3((v1.e.y * v2.e.z - v1.e.z * v2.e.y),
    (-(v1.e.x * v2.e.z - v1.e.z * v2.e.x)),
    (v1.e.x * v2.e.y - v1.e.y * v2.e.x));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3& v)
{
  e.x += v.e.x;
  e.y += v.e.y;
  e.z += v.e.z;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v)
{
  e.x *= v.e.x;
  e.y *= v.e.y;
  e.z *= v.e.z;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v)
{
  e.x /= v.e.x;
  e.y /= v.e.y;
  e.z /= v.e.z;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
  e.x -= v.e.x;
  e.y -= v.e.y;
  e.z -= v.e.z;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
  e.x *= t;
  e.y *= t;
  e.z *= t;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
  float k = 1.0f / t;

  e.x *= k;
  e.y *= k;
  e.z *= k;
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
    vec3 r_out_parallel = -sqrt(float(fabs(1.0 - r_out_perp.squared_length()))) * n;
    return r_out_perp + r_out_parallel;
}