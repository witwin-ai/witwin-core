// Native CUDA triangle-mesh distance / winding / parity kernels.
//
// Direct port of the former ``mesh_sdf.slang`` module. The forward kernels
// reproduce the Slang closest-point (Ericson) and solid-angle math verbatim;
// the backward kernels replace Slang autodiff with hand-derived analytic
// gradients. Every closest-point feature (vertex / edge / face) contributes
// gradients to the triangle vertices through the barycentric weights of the
// closest point, so grad_p == -(grad_a + grad_b + grad_c) holds by
// construction (translation invariance of the distance).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr float kSmoothEps = 1.0e-12f;
constexpr float kDenomEps = 1.0e-12f;
constexpr int kMaxStackDepth = 64;

__host__ __device__ __forceinline__ float3 vsub(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __forceinline__ float3 vadd(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ __forceinline__ float3 vscale(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float vdot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 vcross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float vlen(float3 a) {
  return sqrtf(vdot(a, a));
}

__device__ __forceinline__ float safe_signed_denom(float value, float eps) {
  return value >= 0.0f ? fmaxf(value, eps) : fminf(value, -eps);
}

__device__ __forceinline__ float3 read_point(const float* __restrict__ points, int index) {
  const float* p = points + static_cast<long long>(index) * 3;
  return make_float3(p[0], p[1], p[2]);
}

__device__ __forceinline__ float3 read_vertex(const float* __restrict__ triangles, int triangle, int vertex) {
  const float* t = triangles + (static_cast<long long>(triangle) * 3 + vertex) * 3;
  return make_float3(t[0], t[1], t[2]);
}

__device__ __forceinline__ float3 read_vec3(const float* __restrict__ data, int index) {
  const float* v = data + static_cast<long long>(index) * 3;
  return make_float3(v[0], v[1], v[2]);
}

// Closest point on triangle (a, b, c) to p (Ericson, matching mesh_sdf.slang).
// Returns the squared distance and the barycentric weights (la, lb, lc) of the
// closest point so the backward pass can distribute vertex gradients.
__device__ __forceinline__ float closest_point_barycentric(
    float3 p, float3 a, float3 b, float3 c, float& la, float& lb, float& lc) {
  const float3 ab = vsub(b, a);
  const float3 ac = vsub(c, a);
  const float3 ap = vsub(p, a);

  const float d1 = vdot(ab, ap);
  const float d2 = vdot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) {
    la = 1.0f; lb = 0.0f; lc = 0.0f;
    return vdot(ap, ap);
  }

  const float3 bp = vsub(p, b);
  const float d3 = vdot(ab, bp);
  const float d4 = vdot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) {
    la = 0.0f; lb = 1.0f; lc = 0.0f;
    return vdot(bp, bp);
  }

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = d1 / (d1 - d3);
    la = 1.0f - v; lb = v; lc = 0.0f;
    const float3 delta = vsub(p, vadd(a, vscale(ab, v)));
    return vdot(delta, delta);
  }

  const float3 cp = vsub(p, c);
  const float d5 = vdot(ab, cp);
  const float d6 = vdot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) {
    la = 0.0f; lb = 0.0f; lc = 1.0f;
    return vdot(cp, cp);
  }

  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = d2 / safe_signed_denom(d2 - d6, kDenomEps);
    la = 1.0f - w; lb = 0.0f; lc = w;
    const float3 delta = vsub(p, vadd(a, vscale(ac, w)));
    return vdot(delta, delta);
  }

  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const float w = (d4 - d3) / safe_signed_denom((d4 - d3) + (d5 - d6), kDenomEps);
    la = 0.0f; lb = 1.0f - w; lc = w;
    const float3 bc = vsub(c, b);
    const float3 delta = vsub(p, vadd(b, vscale(bc, w)));
    return vdot(delta, delta);
  }

  const float denom = safe_signed_denom(va + vb + vc, kDenomEps);
  const float v = vb / denom;
  const float w = vc / denom;
  la = 1.0f - v - w; lb = v; lc = w;
  const float3 closest = vadd(vadd(a, vscale(ab, v)), vscale(ac, w));
  const float3 delta = vsub(p, closest);
  return vdot(delta, delta);
}

__device__ __forceinline__ float point_triangle_distance_squared(float3 p, float3 a, float3 b, float3 c) {
  float la, lb, lc;
  return closest_point_barycentric(p, a, b, c, la, lb, lc);
}

__device__ __forceinline__ float triangle_solid_angle(float3 p, float3 a0, float3 b0, float3 c0) {
  const float eps = 1.0e-12f;
  const float3 a = vsub(a0, p);
  const float3 b = vsub(b0, p);
  const float3 c = vsub(c0, p);
  const float la = fmaxf(vlen(a), eps);
  const float lb = fmaxf(vlen(b), eps);
  const float lc = fmaxf(vlen(c), eps);
  const float numerator = vdot(a, vcross(b, c));
  const float denominator =
      la * lb * lc + vdot(a, b) * lc + vdot(b, c) * la + vdot(c, a) * lb;
  return 2.0f * atan2f(numerator, safe_signed_denom(denominator, eps));
}

__device__ __forceinline__ float point_aabb_distance_squared(float3 p, float3 lo, float3 hi) {
  const float dx = p.x < lo.x ? (lo.x - p.x) : (p.x > hi.x ? p.x - hi.x : 0.0f);
  const float dy = p.y < lo.y ? (lo.y - p.y) : (p.y > hi.y ? p.y - hi.y : 0.0f);
  const float dz = p.z < lo.z ? (lo.z - p.z) : (p.z > hi.z ? p.z - hi.z : 0.0f);
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ bool ray_aabb_hit(float3 origin, float3 direction, float3 lo, float3 hi) {
  const float eps = 1.0e-12f;
  const float3 inv = make_float3(
      fabsf(direction.x) > eps ? 1.0f / direction.x : 1.0e30f,
      fabsf(direction.y) > eps ? 1.0f / direction.y : 1.0e30f,
      fabsf(direction.z) > eps ? 1.0f / direction.z : 1.0e30f);
  const float tx1 = (lo.x - origin.x) * inv.x;
  const float tx2 = (hi.x - origin.x) * inv.x;
  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);
  const float ty1 = (lo.y - origin.y) * inv.y;
  const float ty2 = (hi.y - origin.y) * inv.y;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));
  const float tz1 = (lo.z - origin.z) * inv.z;
  const float tz2 = (hi.z - origin.z) * inv.z;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));
  return tmax >= fmaxf(tmin, 0.0f);
}

__device__ __forceinline__ bool ray_triangle_hit(float3 origin, float3 direction, float3 a, float3 b, float3 c) {
  const float epsilon = 1.0e-6f;
  const float3 edge1 = vsub(b, a);
  const float3 edge2 = vsub(c, a);
  const float3 h = vcross(direction, edge2);
  const float det = vdot(edge1, h);
  if (fabsf(det) < epsilon) {
    return false;
  }
  const float inv_det = 1.0f / det;
  const float3 s = vsub(origin, a);
  const float u = inv_det * vdot(s, h);
  if (u < 0.0f || u > 1.0f) {
    return false;
  }
  const float3 q = vcross(s, edge1);
  const float v = inv_det * vdot(direction, q);
  if (v < 0.0f || (u + v) > 1.0f) {
    return false;
  }
  const float t = inv_det * vdot(edge2, q);
  return t > epsilon;
}

// grad(u) where u = sqrt(d2 + kSmoothEps) - const distributes over the triangle
// vertices through the barycentric weights of the closest point. n scales the
// unit direction (p - q) / sqrt(d2 + eps) by the upstream gradient.
__device__ __forceinline__ void accumulate_closest_distance_gradient(
    float3 p, float3 a, float3 b, float3 c, float grad_output,
    float* __restrict__ grad_triangles, int triangle_index, float3& point_gradient) {
  float la, lb, lc;
  const float dist2 = closest_point_barycentric(p, a, b, c, la, lb, lc);
  const float denom = sqrtf(dist2 + kSmoothEps);
  // q = la*a + lb*b + lc*c, so p - q = grad_output * (p - q) / denom = n below.
  const float3 q = vadd(vadd(vscale(a, la), vscale(b, lb)), vscale(c, lc));
  const float3 n = vscale(vsub(p, q), grad_output / denom);
  point_gradient = vadd(point_gradient, n);
  float* g = grad_triangles + static_cast<long long>(triangle_index) * 9;
  atomicAdd(g + 0, -la * n.x);
  atomicAdd(g + 1, -la * n.y);
  atomicAdd(g + 2, -la * n.z);
  atomicAdd(g + 3, -lb * n.x);
  atomicAdd(g + 4, -lb * n.y);
  atomicAdd(g + 5, -lb * n.z);
  atomicAdd(g + 6, -lc * n.x);
  atomicAdd(g + 7, -lc * n.y);
  atomicAdd(g + 8, -lc * n.z);
}

// grad of 2*atan2(N, D) w.r.t. the three (world) vertices, accumulated onto
// grad_triangles; the point gradient is -(gA + gB + gC).
__device__ __forceinline__ void accumulate_solid_angle_gradient(
    float3 p, float3 a0, float3 b0, float3 c0, float grad_output,
    float* __restrict__ grad_triangles, int triangle_index, float3& point_gradient) {
  const float eps = 1.0e-12f;
  const float3 a = vsub(a0, p);
  const float3 b = vsub(b0, p);
  const float3 c = vsub(c0, p);
  const float la = fmaxf(vlen(a), eps);
  const float lb = fmaxf(vlen(b), eps);
  const float lc = fmaxf(vlen(c), eps);
  const float ab = vdot(a, b);
  const float bc = vdot(b, c);
  const float ca = vdot(c, a);
  const float N = vdot(a, vcross(b, c));
  const float D = la * lb * lc + ab * lc + bc * la + ca * lb;
  const float scale = 2.0f * grad_output / fmaxf(N * N + D * D, eps);

  // dN/dA = b x c, dN/dB = c x a, dN/dC = a x b.
  const float3 dN_da = vcross(b, c);
  const float3 dN_db = vcross(c, a);
  const float3 dN_dc = vcross(a, b);
  const float3 a_hat = vscale(a, 1.0f / la);
  const float3 b_hat = vscale(b, 1.0f / lb);
  const float3 c_hat = vscale(c, 1.0f / lc);
  // dD/dA = (lb*lc + b.c) a_hat + lc*b + lb*c, and cyclic variants.
  const float3 dD_da = vadd(vadd(vscale(a_hat, lb * lc + bc), vscale(b, lc)), vscale(c, lb));
  const float3 dD_db = vadd(vadd(vscale(b_hat, la * lc + ca), vscale(a, lc)), vscale(c, la));
  const float3 dD_dc = vadd(vadd(vscale(c_hat, la * lb + ab), vscale(b, la)), vscale(a, lb));

  // d(atan2(N, D)) = (D dN - N dD) / (N^2 + D^2); grad = scale * that.
  const float3 gA = vscale(vsub(vscale(dN_da, D), vscale(dD_da, N)), scale);
  const float3 gB = vscale(vsub(vscale(dN_db, D), vscale(dD_db, N)), scale);
  const float3 gC = vscale(vsub(vscale(dN_dc, D), vscale(dD_dc, N)), scale);

  float* g = grad_triangles + static_cast<long long>(triangle_index) * 9;
  atomicAdd(g + 0, gA.x); atomicAdd(g + 1, gA.y); atomicAdd(g + 2, gA.z);
  atomicAdd(g + 3, gB.x); atomicAdd(g + 4, gB.y); atomicAdd(g + 5, gB.z);
  atomicAdd(g + 6, gC.x); atomicAdd(g + 7, gC.y); atomicAdd(g + 8, gC.z);
  point_gradient = vsub(point_gradient, vadd(vadd(gA, gB), gC));
}

__global__ void query_unsigned_distance_kernel(
    const float* __restrict__ triangles,
    const float* __restrict__ points,
    int triangle_count,
    int point_count,
    float* __restrict__ unsigned_distance,
    int* __restrict__ closest_triangle_index) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float3 p = read_point(points, i);
  float min_dist2 = 1.0e30f;
  int closest = -1;
  for (int t = 0; t < triangle_count; ++t) {
    const float d2 = point_triangle_distance_squared(
        p, read_vertex(triangles, t, 0), read_vertex(triangles, t, 1), read_vertex(triangles, t, 2));
    if (d2 < min_dist2) {
      min_dist2 = d2;
      closest = t;
    }
  }
  unsigned_distance[i] = sqrtf(fmaxf(min_dist2, 0.0f) + kSmoothEps) - sqrtf(kSmoothEps);
  closest_triangle_index[i] = closest;
}

__global__ void query_distance_and_winding_kernel(
    const float* __restrict__ triangles,
    const float* __restrict__ points,
    int triangle_count,
    int point_count,
    float* __restrict__ unsigned_distance,
    float* __restrict__ winding_angle,
    int* __restrict__ closest_triangle_index) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float3 p = read_point(points, i);
  float min_dist2 = 1.0e30f;
  float total_angle = 0.0f;
  int closest = -1;
  for (int t = 0; t < triangle_count; ++t) {
    const float3 a = read_vertex(triangles, t, 0);
    const float3 b = read_vertex(triangles, t, 1);
    const float3 c = read_vertex(triangles, t, 2);
    const float d2 = point_triangle_distance_squared(p, a, b, c);
    if (d2 < min_dist2) {
      min_dist2 = d2;
      closest = t;
    }
    total_angle += triangle_solid_angle(p, a, b, c);
  }
  unsigned_distance[i] = sqrtf(fmaxf(min_dist2, 0.0f) + kSmoothEps) - sqrtf(kSmoothEps);
  winding_angle[i] = total_angle;
  closest_triangle_index[i] = closest;
}

__global__ void query_unsigned_distance_bvh_kernel(
    const float* __restrict__ triangles,
    const float* __restrict__ points,
    const float* __restrict__ node_bbox_min,
    const float* __restrict__ node_bbox_max,
    const int* __restrict__ node_left,
    const int* __restrict__ node_right,
    const int* __restrict__ node_start,
    const int* __restrict__ node_count,
    const int* __restrict__ triangle_indices,
    int point_count,
    float* __restrict__ unsigned_distance,
    int* __restrict__ closest_triangle_index) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  int stack[kMaxStackDepth];
  int stack_size = 0;
  stack[stack_size++] = 0;

  const float3 p = read_point(points, i);
  float min_dist2 = 1.0e30f;
  int closest = -1;

  while (stack_size > 0) {
    const int node = stack[--stack_size];
    const float node_dist2 = point_aabb_distance_squared(
        p, read_vec3(node_bbox_min, node), read_vec3(node_bbox_max, node));
    if (node_dist2 > min_dist2) {
      continue;
    }
    const int count = node_count[node];
    if (count > 0) {
      const int start = node_start[node];
      for (int offset = 0; offset < count; ++offset) {
        const int t = triangle_indices[start + offset];
        const float d2 = point_triangle_distance_squared(
            p, read_vertex(triangles, t, 0), read_vertex(triangles, t, 1), read_vertex(triangles, t, 2));
        if (d2 < min_dist2) {
          min_dist2 = d2;
          closest = t;
        }
      }
      continue;
    }
    const int left = node_left[node];
    const int right = node_right[node];
    if (left < 0 || right < 0) {
      continue;
    }
    const float left_dist2 = point_aabb_distance_squared(
        p, read_vec3(node_bbox_min, left), read_vec3(node_bbox_max, left));
    const float right_dist2 = point_aabb_distance_squared(
        p, read_vec3(node_bbox_min, right), read_vec3(node_bbox_max, right));
    if (left_dist2 < right_dist2) {
      if (stack_size + 2 <= kMaxStackDepth) {
        stack[stack_size++] = right;
        stack[stack_size++] = left;
      }
    } else {
      if (stack_size + 2 <= kMaxStackDepth) {
        stack[stack_size++] = left;
        stack[stack_size++] = right;
      }
    }
  }

  unsigned_distance[i] = sqrtf(fmaxf(min_dist2, 0.0f) + kSmoothEps) - sqrtf(kSmoothEps);
  closest_triangle_index[i] = closest;
}

__global__ void query_parity_sign_bvh_kernel(
    const float* __restrict__ triangles,
    const float* __restrict__ points,
    const float* __restrict__ node_bbox_min,
    const float* __restrict__ node_bbox_max,
    const int* __restrict__ node_left,
    const int* __restrict__ node_right,
    const int* __restrict__ node_start,
    const int* __restrict__ node_count,
    const int* __restrict__ triangle_indices,
    float jitter_scale,
    int point_count,
    int* __restrict__ inside) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  int stack[kMaxStackDepth];
  int stack_size = 0;
  stack[stack_size++] = 0;

  const float3 p = read_point(points, i);
  const float3 direction = make_float3(1.0f, 0.0f, 0.0f);
  const float3 origin = make_float3(p.x, p.y + jitter_scale, p.z + jitter_scale * 1.618034f);
  int intersections = 0;

  while (stack_size > 0) {
    const int node = stack[--stack_size];
    if (!ray_aabb_hit(origin, direction, read_vec3(node_bbox_min, node), read_vec3(node_bbox_max, node))) {
      continue;
    }
    const int count = node_count[node];
    if (count > 0) {
      const int start = node_start[node];
      for (int offset = 0; offset < count; ++offset) {
        const int t = triangle_indices[start + offset];
        if (ray_triangle_hit(origin, direction,
                             read_vertex(triangles, t, 0), read_vertex(triangles, t, 1), read_vertex(triangles, t, 2))) {
          intersections += 1;
        }
      }
      continue;
    }
    const int left = node_left[node];
    const int right = node_right[node];
    if (left >= 0 && stack_size < kMaxStackDepth) {
      stack[stack_size++] = left;
    }
    if (right >= 0 && stack_size < kMaxStackDepth) {
      stack[stack_size++] = right;
    }
  }

  inside[i] = (intersections & 1) != 0 ? 1 : 0;
}

__global__ void backward_unsigned_distance_kernel(
    const float* __restrict__ triangles,
    const float* __restrict__ points,
    const int* __restrict__ closest_triangle_index,
    const float* __restrict__ grad_unsigned_distance,
    int point_count,
    float* __restrict__ grad_triangles,
    float* __restrict__ grad_points) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float grad_output = grad_unsigned_distance[i];
  if (grad_output == 0.0f) {
    return;
  }
  const int t = closest_triangle_index[i];
  if (t < 0) {
    return;
  }
  const float3 p = read_point(points, i);
  float3 point_gradient = make_float3(0.0f, 0.0f, 0.0f);
  accumulate_closest_distance_gradient(
      p, read_vertex(triangles, t, 0), read_vertex(triangles, t, 1), read_vertex(triangles, t, 2),
      grad_output, grad_triangles, t, point_gradient);
  float* gp = grad_points + static_cast<long long>(i) * 3;
  gp[0] = point_gradient.x;
  gp[1] = point_gradient.y;
  gp[2] = point_gradient.z;
}

__global__ void backward_distance_and_winding_kernel(
    const float* __restrict__ triangles,
    const float* __restrict__ points,
    const int* __restrict__ closest_triangle_index,
    const float* __restrict__ grad_unsigned_distance,
    const float* __restrict__ grad_winding_angle,
    int triangle_count,
    int point_count,
    float* __restrict__ grad_triangles,
    float* __restrict__ grad_points) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= point_count) {
    return;
  }
  const float grad_unsigned = grad_unsigned_distance[i];
  const float grad_angle = grad_winding_angle[i];
  if (grad_unsigned == 0.0f && grad_angle == 0.0f) {
    return;
  }
  const float3 p = read_point(points, i);
  float3 point_gradient = make_float3(0.0f, 0.0f, 0.0f);

  if (grad_angle != 0.0f) {
    for (int t = 0; t < triangle_count; ++t) {
      accumulate_solid_angle_gradient(
          p, read_vertex(triangles, t, 0), read_vertex(triangles, t, 1), read_vertex(triangles, t, 2),
          grad_angle, grad_triangles, t, point_gradient);
    }
  }

  if (grad_unsigned != 0.0f) {
    const int t = closest_triangle_index[i];
    if (t >= 0) {
      accumulate_closest_distance_gradient(
          p, read_vertex(triangles, t, 0), read_vertex(triangles, t, 1), read_vertex(triangles, t, 2),
          grad_unsigned, grad_triangles, t, point_gradient);
    }
  }

  float* gp = grad_points + static_cast<long long>(i) * 3;
  gp[0] = point_gradient.x;
  gp[1] = point_gradient.y;
  gp[2] = point_gradient.z;
}

dim3 linear_grid(int elements, int block_size) {
  return dim3(static_cast<unsigned int>((elements + block_size - 1) / block_size), 1, 1);
}

void check_f32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

constexpr int kBlock = 256;

}  // namespace

void query_mesh_unsigned_distance_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor unsigned_distance, at::Tensor closest_triangle_index) {
  check_f32(triangles, "triangles");
  check_f32(points, "points");
  check_f32(unsigned_distance, "unsigned_distance");
  check_i32(closest_triangle_index, "closest_triangle_index");
  const c10::cuda::CUDAGuard guard(points.device());
  const int point_count = static_cast<int>(points.size(0));
  if (point_count == 0) {
    return;
  }
  query_unsigned_distance_kernel<<<linear_grid(point_count, kBlock), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(), points.data_ptr<float>(),
      static_cast<int>(triangles.size(0)), point_count,
      unsigned_distance.data_ptr<float>(), closest_triangle_index.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void query_mesh_distance_and_winding_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor unsigned_distance,
    at::Tensor winding_angle, at::Tensor closest_triangle_index) {
  check_f32(triangles, "triangles");
  check_f32(points, "points");
  check_f32(unsigned_distance, "unsigned_distance");
  check_f32(winding_angle, "winding_angle");
  check_i32(closest_triangle_index, "closest_triangle_index");
  const c10::cuda::CUDAGuard guard(points.device());
  const int point_count = static_cast<int>(points.size(0));
  if (point_count == 0) {
    return;
  }
  query_distance_and_winding_kernel<<<linear_grid(point_count, kBlock), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(), points.data_ptr<float>(),
      static_cast<int>(triangles.size(0)), point_count,
      unsigned_distance.data_ptr<float>(), winding_angle.data_ptr<float>(),
      closest_triangle_index.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void query_mesh_unsigned_distance_bvh_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor node_bbox_min, at::Tensor node_bbox_max,
    at::Tensor node_left, at::Tensor node_right, at::Tensor node_start, at::Tensor node_count,
    at::Tensor triangle_indices, at::Tensor unsigned_distance, at::Tensor closest_triangle_index) {
  check_f32(triangles, "triangles");
  check_f32(points, "points");
  check_f32(node_bbox_min, "node_bbox_min");
  check_f32(node_bbox_max, "node_bbox_max");
  check_i32(node_left, "node_left");
  check_i32(node_right, "node_right");
  check_i32(node_start, "node_start");
  check_i32(node_count, "node_count");
  check_i32(triangle_indices, "triangle_indices");
  check_f32(unsigned_distance, "unsigned_distance");
  check_i32(closest_triangle_index, "closest_triangle_index");
  const c10::cuda::CUDAGuard guard(points.device());
  const int point_count = static_cast<int>(points.size(0));
  if (point_count == 0) {
    return;
  }
  query_unsigned_distance_bvh_kernel<<<linear_grid(point_count, kBlock), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(), points.data_ptr<float>(),
      node_bbox_min.data_ptr<float>(), node_bbox_max.data_ptr<float>(),
      node_left.data_ptr<int>(), node_right.data_ptr<int>(),
      node_start.data_ptr<int>(), node_count.data_ptr<int>(),
      triangle_indices.data_ptr<int>(), point_count,
      unsigned_distance.data_ptr<float>(), closest_triangle_index.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void query_mesh_parity_sign_bvh_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor node_bbox_min, at::Tensor node_bbox_max,
    at::Tensor node_left, at::Tensor node_right, at::Tensor node_start, at::Tensor node_count,
    at::Tensor triangle_indices, double jitter_scale, at::Tensor inside) {
  check_f32(triangles, "triangles");
  check_f32(points, "points");
  check_f32(node_bbox_min, "node_bbox_min");
  check_f32(node_bbox_max, "node_bbox_max");
  check_i32(node_left, "node_left");
  check_i32(node_right, "node_right");
  check_i32(node_start, "node_start");
  check_i32(node_count, "node_count");
  check_i32(triangle_indices, "triangle_indices");
  check_i32(inside, "inside");
  const c10::cuda::CUDAGuard guard(points.device());
  const int point_count = static_cast<int>(points.size(0));
  if (point_count == 0) {
    return;
  }
  query_parity_sign_bvh_kernel<<<linear_grid(point_count, kBlock), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(), points.data_ptr<float>(),
      node_bbox_min.data_ptr<float>(), node_bbox_max.data_ptr<float>(),
      node_left.data_ptr<int>(), node_right.data_ptr<int>(),
      node_start.data_ptr<int>(), node_count.data_ptr<int>(),
      triangle_indices.data_ptr<int>(), static_cast<float>(jitter_scale), point_count,
      inside.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_mesh_unsigned_distance_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor closest_triangle_index,
    at::Tensor grad_unsigned_distance, at::Tensor grad_triangles, at::Tensor grad_points) {
  check_f32(triangles, "triangles");
  check_f32(points, "points");
  check_i32(closest_triangle_index, "closest_triangle_index");
  check_f32(grad_unsigned_distance, "grad_unsigned_distance");
  check_f32(grad_triangles, "grad_triangles");
  check_f32(grad_points, "grad_points");
  const c10::cuda::CUDAGuard guard(points.device());
  const int point_count = static_cast<int>(points.size(0));
  if (point_count == 0) {
    return;
  }
  backward_unsigned_distance_kernel<<<linear_grid(point_count, kBlock), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(), points.data_ptr<float>(),
      closest_triangle_index.data_ptr<int>(), grad_unsigned_distance.data_ptr<float>(),
      point_count, grad_triangles.data_ptr<float>(), grad_points.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_mesh_distance_and_winding_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor closest_triangle_index,
    at::Tensor grad_unsigned_distance, at::Tensor grad_winding_angle,
    at::Tensor grad_triangles, at::Tensor grad_points) {
  check_f32(triangles, "triangles");
  check_f32(points, "points");
  check_i32(closest_triangle_index, "closest_triangle_index");
  check_f32(grad_unsigned_distance, "grad_unsigned_distance");
  check_f32(grad_winding_angle, "grad_winding_angle");
  check_f32(grad_triangles, "grad_triangles");
  check_f32(grad_points, "grad_points");
  const c10::cuda::CUDAGuard guard(points.device());
  const int point_count = static_cast<int>(points.size(0));
  if (point_count == 0) {
    return;
  }
  backward_distance_and_winding_kernel<<<linear_grid(point_count, kBlock), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
      triangles.data_ptr<float>(), points.data_ptr<float>(),
      closest_triangle_index.data_ptr<int>(), grad_unsigned_distance.data_ptr<float>(),
      grad_winding_angle.data_ptr<float>(), static_cast<int>(triangles.size(0)), point_count,
      grad_triangles.data_ptr<float>(), grad_points.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
