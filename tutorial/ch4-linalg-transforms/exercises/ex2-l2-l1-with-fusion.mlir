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
    // TODO: Implement L2/L1 tiling with fusion
    //
    // Steps:
    // 1. Match the add (consumer) - it has only parallel iterators
    // 2. Tile add with L2 sizes [64, 64] using transform.structured.tile_using_forall
    // 3. Match and fuse matmul (producer) into the forall
    // 4. Match and fuse fill (matmul's producer) into the forall
    // 5. Tile reduction dimension of matmul with [0, 0, 32]
    // 6. Tile matmul with L1 sizes [16, 16, 8]
    // 7. Tile add with L1 sizes [16, 16]
    //
    // Hints:
    // - Tile the CONSUMER first, then fuse producers
    // - Use transform.structured.tile_using_forall for parallel tiling
    // - Use transform.structured.fuse_into_containing_op for fusion
    // - See examples/03-tile-and-fuse.mlir for reference

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
