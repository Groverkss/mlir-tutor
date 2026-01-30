// Example 3: Multi-level Tiling and Fusion
//
// This example shows a complete tiling and fusion pipeline for matmul + bias add.
//
// RUN: tutorial-opt --transform-interpreter %s

#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>
#map_elem = affine_map<(i, j) -> (i, j)>

func.func @matmul_bias(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>,
                       %bias: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // Allocate output tensor
  %empty = tensor.empty() : tensor<64x64xf32>

  // Initialize to zeros (producer of matmul)
  %zero = arith.constant 0.0 : f32
  %C_init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>)
      -> tensor<64x64xf32>

  // Matmul: temp = A @ B (producer of add)
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

  // Add bias: result = matmul + bias (consumer)
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
    // 1. Find the add (consumer) - it has only parallel iterators
    %add = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // 2. Tile add with scf.forall (parallel tiling)
    %tiled_add, %forall = transform.structured.tile_using_forall %add
        tile_sizes [16, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 3. Find matmul (producer of add) - it has a reduction iterator
    %matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // 4. Fuse matmul into the forall
    %fused_matmul, %new_forall = transform.structured.fuse_into_containing_op
        %matmul into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 5. Find and fuse fill (producer of matmul)
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused_fill, %final_forall = transform.structured.fuse_into_containing_op
        %fill into %new_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 6. Tile reduction dimension of the fused matmul
    %inner_matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %final_forall
        : (!transform.any_op) -> !transform.any_op
    %tiled_k, %k_loop = transform.structured.tile_using_for %inner_matmul
        tile_sizes [0, 0, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// After transformation:
//
// scf.forall (%i, %j) in (4, 4) {  // 64/16 = 4 tiles per dimension
//   // Fill is fused: initialize 16x16 tile
//   %c_tile = linalg.fill ...
//
//   // Matmul is fused: compute 16x16 output tile
//   // Reduction is tiled: iterate over k in steps of 8
//   scf.for %k = 0 to 128 step 8 {
//     %a_tile = tensor.extract_slice %A[%i*16, %k] [16, 8] [1, 1]
//     %b_tile = tensor.extract_slice %B[%k, %j*16] [8, 16] [1, 1]
//     %matmul_tile = linalg.generic ... ins(%a_tile, %b_tile) outs(%c_tile)
//   }
//
//   // Add is tiled: add bias to 16x16 tile
//   %result_tile = linalg.generic ... ins(%matmul_tile, %bias_tile)
//
//   // Insert result
//   tensor.parallel_insert_slice %result_tile into %out[%i*16, %j*16]
// }
