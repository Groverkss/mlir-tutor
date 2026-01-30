// Example 4: Vectorization
//
// This example extends example 3 by adding vectorization after tiling and fusion.
//
// RUN: tutorial-opt --transform-interpreter %s

#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>
#map_elem = affine_map<(i, j) -> (i, j)>

func.func @matmul_bias(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>,
                       %bias: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>)
      -> tensor<64x64xf32>

  %matmul = linalg.generic {
    indexing_maps = [#map_A, #map_B, #map_C],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
    outs(%C_init : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
  } -> tensor<64x64xf32>

  %result = linalg.generic {
    indexing_maps = [#map_elem, #map_elem, #map_elem],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%empty : tensor<64x64xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      linalg.yield %sum : f32
  } -> tensor<64x64xf32>

  return %result : tensor<64x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // ========== Same tiling/fusion as example 3 ==========

    // 1. Find and tile the add (consumer)
    %add = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %tiled_add, %forall = transform.structured.tile_using_forall %add
        tile_sizes [16, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 2. Fuse matmul into forall
    %matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused_matmul, %new_forall = transform.structured.fuse_into_containing_op
        %matmul into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. Fuse fill into forall
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused_fill, %final_forall = transform.structured.fuse_into_containing_op
        %fill into %new_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4. Tile reduction dimension
    %inner_matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %final_forall
        : (!transform.any_op) -> !transform.any_op
    %tiled_k, %k_loop = transform.structured.tile_using_for %inner_matmul
        tile_sizes [0, 0, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ========== NEW: Vectorization ==========

    // 5. Vectorize all linalg operations in the function
    // This converts linalg ops to vector operations (vector.transfer_read,
    // vector.contract, vector.transfer_write, etc.)
    %func = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

// After vectorization, the innermost matmul becomes:
//
// %a_vec = vector.transfer_read %A_tile[...] : tensor<...> -> vector<16x8xf32>
// %b_vec = vector.transfer_read %B_tile[...] : tensor<...> -> vector<8x16xf32>
// %c_vec = vector.transfer_read %C_tile[...] : tensor<...> -> vector<16x16xf32>
// %result = vector.contract %a_vec, %b_vec, %c_vec : ... -> vector<16x16xf32>
// vector.transfer_write %result, %C_tile[...] : vector<16x16xf32>
