// Exercise 2: L2/L1 Tiling With Fusion
//
// This example shows two-level tiling with producer-consumer fusion.
// Key insight: Tile the CONSUMER first, then fuse the producer into it.
//
// Pipeline:
// 1. Tile add (consumer) with L2 sizes [64, 64] using scf.forall
// 2. Fuse matmul (producer) into the forall
// 3. Fuse fill into the forall
// 4. Tile reduction dimension of matmul with L2 size [32]
// 5. Tile the inner matmul with L1 sizes [16, 16, 8]
// 6. Tile add with L1 sizes [16, 16]
//
// RUN: tutorial-opt --transform-interpreter %s

#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>
#map_elem = affine_map<(i, j) -> (i, j)>

func.func @matmul_bias(%A: tensor<128x256xf32>, %B: tensor<256x128xf32>,
                       %bias: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %empty = tensor.empty() : tensor<128x128xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x128xf32>)
      -> tensor<128x128xf32>

  %matmul = linalg.generic {
    indexing_maps = [#map_A, #map_B, #map_C],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
    outs(%C_init : tensor<128x128xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
  } -> tensor<128x128xf32>

  %result = linalg.generic {
    indexing_maps = [#map_elem, #map_elem, #map_elem],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%empty : tensor<128x128xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      linalg.yield %sum : f32
  } -> tensor<128x128xf32>

  return %result : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Step 1: Find the add (consumer) - it has only parallel iterators
    %add = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Step 2: Tile add with L2 sizes [64, 64] using scf.forall (parallel)
    %tiled_add, %forall = transform.structured.tile_using_forall %add
        tile_sizes [64, 64]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3: Find matmul (producer of add)
    %matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Step 4: Fuse matmul into the forall
    %fused_matmul, %new_forall = transform.structured.fuse_into_containing_op
        %matmul into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 5: Find and fuse fill (producer of matmul)
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused_fill, %final_forall = transform.structured.fuse_into_containing_op
        %fill into %new_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 6: Tile reduction dimension of the fused matmul with L2 size [32]
    %inner_matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %final_forall
        : (!transform.any_op) -> !transform.any_op
    %matmul_k_tiled, %k_loop = transform.structured.tile_using_for %inner_matmul
        tile_sizes [0, 0, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 7: Tile matmul with L1 sizes [16, 16, 8]
    %matmul_for_l1 = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %k_loop
        : (!transform.any_op) -> !transform.any_op
    %matmul_l1, %m_l1, %n_l1, %k_l1 = transform.structured.tile_using_for %matmul_for_l1
        tile_sizes [16, 16, 8]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 8: Tile add with L1 sizes [16, 16]
    %add_for_l1 = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>]} in %final_forall
        : (!transform.any_op) -> !transform.any_op
    %add_l1, %i_l1, %j_l1 = transform.structured.tile_using_for %add_for_l1
        tile_sizes [16, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}

// Result structure (with fusion):
//
// scf.forall (%i_l2, %j_l2) in (2, 2) {  // 128/64 = 2 tiles, parallel L2
//
//   // Fill is fused: initialize 64x64 tile
//   %c_tile = linalg.fill ...
//
//   // Matmul is fused, with k tiled at L2
//   scf.for %k_l2 = 0 to 256 step 32 {
//     // L1 tiling of matmul
//     scf.for %m_l1 = 0 to 64 step 16 {
//       scf.for %n_l1 = 0 to 64 step 16 {
//         scf.for %k_l1 = 0 to 32 step 8 {
//           linalg.generic (matmul on 16x16x8 tile)
//         }
//       }
//     }
//   }
//
//   // Add is tiled at L1
//   scf.for %i_l1 = 0 to 64 step 16 {
//     scf.for %j_l1 = 0 to 64 step 16 {
//       linalg.generic (add on 16x16 tile)
//     }
//   }
//
//   tensor.parallel_insert_slice ...
// }
//
// Benefits of fusion:
// - The matmul output (64x64 = 16KB) fits in L2 cache
// - The add consumes it immediately while it's still hot
// - No need to write/read the full 128x128 intermediate result to memory
