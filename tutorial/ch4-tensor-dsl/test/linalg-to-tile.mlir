// RUN: tutorial-opt %s -linalg-to-tiny-tile | FileCheck %s
// Test conversion of linalg.generic elementwise operations to tiny_tile.

#map = affine_map<(i, j) -> (i, j)>

// CHECK-LABEL: func.func @elementwise_add
func.func @elementwise_add(%a: tensor<64x128xf16>, %b: tensor<64x128xf16>,
                           %c: tensor<64x128xf16>) -> tensor<64x128xf16> {
  // CHECK: %[[A_PTR:.*]] = tiny_tensor.to_ptr %arg0
  // CHECK: %[[B_PTR:.*]] = tiny_tensor.to_ptr %arg1
  // CHECK: %[[C_PTR:.*]] = tiny_tensor.to_ptr %arg2
  // CHECK: %[[TILE_A:.*]] = tiny_tile.load %[[A_PTR]]
  // CHECK: %[[TILE_B:.*]] = tiny_tile.load %[[B_PTR]]
  // CHECK: %[[TILE_SUM:.*]] = tiny_tile.elementwise add %[[TILE_A]], %[[TILE_B]]
  // CHECK: tiny_tile.store %[[TILE_SUM]], %[[C_PTR]]
  // CHECK: %[[RESULT:.*]] = tiny_tensor.from_ptr %[[C_PTR]]
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<64x128xf16>, tensor<64x128xf16>)
    outs(%c : tensor<64x128xf16>) {
  ^bb0(%a_elem: f16, %b_elem: f16, %c_elem: f16):
    %sum = arith.addf %a_elem, %b_elem : f16
    linalg.yield %sum : f16
  } -> tensor<64x128xf16>
  // CHECK: return %[[RESULT]]
  return %result : tensor<64x128xf16>
}

// CHECK-LABEL: func.func @elementwise_mul
func.func @elementwise_mul(%a: tensor<32x64xf16>, %b: tensor<32x64xf16>,
                           %c: tensor<32x64xf16>) -> tensor<32x64xf16> {
  // CHECK: tiny_tile.elementwise mul
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<32x64xf16>, tensor<32x64xf16>)
    outs(%c : tensor<32x64xf16>) {
  ^bb0(%a_elem: f16, %b_elem: f16, %c_elem: f16):
    %prod = arith.mulf %a_elem, %b_elem : f16
    linalg.yield %prod : f16
  } -> tensor<32x64xf16>
  return %result : tensor<32x64xf16>
}

// CHECK-LABEL: func.func @elementwise_chain
func.func @elementwise_chain(%a: tensor<64x128xf16>, %b: tensor<64x128xf16>,
                              %c: tensor<64x128xf16>) -> tensor<64x128xf16> {
  // Test chained operations: (a + b) * c
  // First: a + b
  // CHECK: tiny_tile.elementwise add
  %sum = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<64x128xf16>, tensor<64x128xf16>)
    outs(%c : tensor<64x128xf16>) {
  ^bb0(%a_elem: f16, %b_elem: f16, %c_elem: f16):
    %s = arith.addf %a_elem, %b_elem : f16
    linalg.yield %s : f16
  } -> tensor<64x128xf16>

  // Then: sum * c
  // CHECK: tiny_tile.elementwise mul
  %result = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%sum, %c : tensor<64x128xf16>, tensor<64x128xf16>)
    outs(%c : tensor<64x128xf16>) {
  ^bb0(%s_elem: f16, %c_elem: f16, %out_elem: f16):
    %p = arith.mulf %s_elem, %c_elem : f16
    linalg.yield %p : f16
  } -> tensor<64x128xf16>

  return %result : tensor<64x128xf16>
}
