// Test for transform.tiny.tile_to_l2_l1
//
// This test verifies that the two-level tiling transform op works correctly.
// It should create nested loops for L2 and L1 tile sizes.
//
// RUN: tutorial-opt --transform-interpreter %s | FileCheck %s

#map = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>

func.func @matmul(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>,
                  %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Two-level tiling: L2 [32, 32, 16] then L1 [8, 8, 4]
    %tiled = transform.tiny.tile_to_l2_l1 %matmul
        l2 [32, 32, 16] l1 [8, 8, 4]
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

// CHECK: func.func @matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.generic
