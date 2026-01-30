// Example 2: Tiling a Single Operation
//
// This example shows how to tile a linalg operation using the transform dialect.
//
// RUN: tutorial-opt --transform-interpreter %s

#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>

func.func @matmul(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>,
                  %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.generic {
    indexing_maps = [#map_A, #map_B, #map_C],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
    outs(%C : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
  } -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// Transform script: tile the matmul with tile sizes [32, 32, 16]
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Find the linalg.generic operation (our matmul)
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Tile with sizes [32, 32, 16] for m, n, k dimensions
    // This creates 3 nested scf.for loops
    %tiled, %loop_m, %loop_n, %loop_k = transform.structured.tile_using_for %matmul
        tile_sizes [32, 32, 16]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}

// After tiling, the computation becomes:
//
// scf.for %m = 0 to 64 step 32 {
//   scf.for %n = 0 to 64 step 32 {
//     scf.for %k = 0 to 128 step 16 {
//       // Extract 32x16 tile from A
//       %a_tile = tensor.extract_slice %A[%m, %k] [32, 16] [1, 1]
//       // Extract 16x32 tile from B
//       %b_tile = tensor.extract_slice %B[%k, %n] [16, 32] [1, 1]
//       // Extract 32x32 tile from C
//       %c_tile = tensor.extract_slice %C[%m, %n] [32, 32] [1, 1]
//       // Compute on tiles
//       %result_tile = linalg.generic ... ins(%a_tile, %b_tile) outs(%c_tile)
//       // Insert result back
//       tensor.insert_slice %result_tile into %C[%m, %n] [32, 32] [1, 1]
//     }
//   }
// }
