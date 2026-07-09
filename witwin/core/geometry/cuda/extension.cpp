#include <torch/extension.h>
#include <torch/cuda.h>

bool is_available() {
  return torch::cuda::is_available();
}

void query_mesh_unsigned_distance_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor unsigned_distance, at::Tensor closest_triangle_index);
void query_mesh_distance_and_winding_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor unsigned_distance,
    at::Tensor winding_angle, at::Tensor closest_triangle_index);
void query_mesh_unsigned_distance_bvh_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor node_bbox_min, at::Tensor node_bbox_max,
    at::Tensor node_left, at::Tensor node_right, at::Tensor node_start, at::Tensor node_count,
    at::Tensor triangle_indices, at::Tensor unsigned_distance, at::Tensor closest_triangle_index);
void query_mesh_parity_sign_bvh_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor node_bbox_min, at::Tensor node_bbox_max,
    at::Tensor node_left, at::Tensor node_right, at::Tensor node_start, at::Tensor node_count,
    at::Tensor triangle_indices, double jitter_scale, at::Tensor inside);
void backward_mesh_unsigned_distance_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor closest_triangle_index,
    at::Tensor grad_unsigned_distance, at::Tensor grad_triangles, at::Tensor grad_points);
void backward_mesh_distance_and_winding_cuda(
    at::Tensor triangles, at::Tensor points, at::Tensor closest_triangle_index,
    at::Tensor grad_unsigned_distance, at::Tensor grad_winding_angle,
    at::Tensor grad_triangles, at::Tensor grad_points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_available", &is_available, "Return whether CUDA is available to PyTorch.");
  m.def("query_mesh_unsigned_distance", &query_mesh_unsigned_distance_cuda,
        "Brute-force unsigned point-to-mesh distance and closest triangle index.");
  m.def("query_mesh_distance_and_winding", &query_mesh_distance_and_winding_cuda,
        "Brute-force unsigned distance plus solid-angle winding sum.");
  m.def("query_mesh_unsigned_distance_bvh", &query_mesh_unsigned_distance_bvh_cuda,
        "BVH-accelerated unsigned distance and closest triangle index.");
  m.def("query_mesh_parity_sign_bvh", &query_mesh_parity_sign_bvh_cuda,
        "BVH-accelerated ray-parity inside/outside test.");
  m.def("backward_mesh_unsigned_distance", &backward_mesh_unsigned_distance_cuda,
        "Backward pass for the unsigned distance query.");
  m.def("backward_mesh_distance_and_winding", &backward_mesh_distance_and_winding_cuda,
        "Backward pass for the distance-and-winding query.");
}
