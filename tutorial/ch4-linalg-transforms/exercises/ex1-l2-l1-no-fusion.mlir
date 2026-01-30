// Exercise 1: L2/L1 Tiling Without Fusion
//
// This example shows two-level cache tiling for matmul and add, tiled separately.
// This demonstrates hierarchical tiling for cache locality but without fusion.
//
// Pipeline:
// 1. Tile matmul with L2 sizes [64, 64, 32] using scf.for
// 2. Tile matmul again with L1 sizes [16, 16, 8] using scf.for
// 3. Tile add with L2 sizes [64, 64] using scf.for
// 4. Tile add again with L1 sizes [16, 16] using scf.for
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
    // TODO: Implement L2/L1 tiling without fusion
    //
    // Steps:
    // 1. Match the matmul (has reduction iterator)
    // 2. Tile matmul with L2 sizes [64, 64, 32] using transform.structured.tile_using_for
    // 3. Tile matmul again with L1 sizes [16, 16, 8]
    // 4. Match the add (only parallel iterators)
    // 5. Tile add with L2 sizes [64, 64]
    // 6. Tile add again with L1 sizes [16, 16]
    //
    // Hints:
    // - Use transform.structured.match to find operations
    // - Use transform.structured.tile_using_for for tiling
    // - See examples/02-tile-single-op.mlir for reference

    transform.yield
  }
}

// Result structure (without fusion):
//
// scf.for %m_l2 = 0 to 128 step 64 {           // L2 m
//   scf.for %n_l2 = 0 to 128 step 64 {         // L2 n
//     scf.for %k_l2 = 0 to 256 step 32 {       // L2 k
//       scf.for %m_l1 = 0 to 64 step 16 {      // L1 m
//         scf.for %n_l1 = 0 to 64 step 16 {    // L1 n
//           scf.for %k_l1 = 0 to 32 step 8 {   // L1 k
//             linalg.generic (matmul on 16x16x8 tile)
//           }
//         }
//       }
//     }
//   }
// }
//
// // Separate loops for add (not fused!)
// scf.for %i_l2 = 0 to 128 step 64 {           // L2 i
//   scf.for %j_l2 = 0 to 128 step 64 {         // L2 j
//     scf.for %i_l1 = 0 to 64 step 16 {        // L1 i
//       scf.for %j_l1 = 0 to 64 step 16 {      // L1 j
//         linalg.generic (add on 16x16 tile)
//       }
//     }
//   }
// }
//
// Note: Without fusion, the matmul result tensor must be fully materialized
// before the add can begin. This hurts cache locality since the intermediate
// result doesn't fit in cache.
