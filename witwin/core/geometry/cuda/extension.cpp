#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(witwin_core_mesh_sdf_cuda, m) {
  m.def(
      "query_mesh_unsigned_distance(Tensor triangles, Tensor points, "
      "Tensor(a!) unsigned_distance, Tensor(b!) closest_triangle_index) -> ()");
  m.def(
      "query_mesh_distance_and_winding(Tensor triangles, Tensor points, "
      "Tensor(a!) unsigned_distance, Tensor(b!) winding_angle, "
      "Tensor(c!) closest_triangle_index) -> ()");
  m.def(
      "query_mesh_unsigned_distance_bvh(Tensor triangles, Tensor points, "
      "Tensor node_bbox_min, Tensor node_bbox_max, Tensor node_left, Tensor node_right, "
      "Tensor node_start, Tensor node_count, Tensor triangle_indices, "
      "Tensor(a!) unsigned_distance, Tensor(b!) closest_triangle_index) -> ()");
  m.def(
      "query_mesh_parity_sign_bvh(Tensor triangles, Tensor points, "
      "Tensor node_bbox_min, Tensor node_bbox_max, Tensor node_left, Tensor node_right, "
      "Tensor node_start, Tensor node_count, Tensor triangle_indices, float jitter_scale, "
      "Tensor(a!) inside) -> ()");
  m.def(
      "backward_mesh_unsigned_distance(Tensor triangles, Tensor points, "
      "Tensor closest_triangle_index, Tensor grad_unsigned_distance, "
      "Tensor(a!) grad_triangles, Tensor(b!) grad_points) -> ()");
  m.def(
      "backward_mesh_distance_and_winding(Tensor triangles, Tensor points, "
      "Tensor closest_triangle_index, Tensor grad_unsigned_distance, Tensor grad_winding_angle, "
      "Tensor(a!) grad_triangles, Tensor(b!) grad_points) -> ()");
}
