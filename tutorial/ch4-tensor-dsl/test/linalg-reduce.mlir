// RUN: tutorial-opt %s -linalg-to-tiny-tile | FileCheck %s
// Test conversion of linalg.generic reduction operations to tiny_tile.

#map_input = affine_map<(i, j) -> (i, j)>
#map_output = affine_map<(i, j) -> ()>

// CHECK-LABEL: func.func @full_reduction_sum
func.func @full_reduction_sum(%input: tensor<64x128xf16>,
                               %init: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[PTR:.*]] = tiny_tensor.to_ptr %arg0
  // CHECK: %[[TILE:.*]] = tiny_tile.load %[[PTR]]
  // CHECK: %[[SUM:.*]] = tiny_tile.reduce %[[TILE]]
  // CHECK: %[[RESULT:.*]] = tensor.from_elements %[[SUM]]
  %result = linalg.generic {
    indexing_maps = [#map_input, #map_output],
    iterator_types = ["reduction", "reduction"]
  } ins(%input : tensor<64x128xf16>)
    outs(%init : tensor<f16>) {
  ^bb0(%in: f16, %out: f16):
    %sum = arith.addf %in, %out : f16
    linalg.yield %sum : f16
  } -> tensor<f16>
  // CHECK: return %[[RESULT]]
  return %result : tensor<f16>
}

// CHECK-LABEL: func.func @reduction_small_tile
func.func @reduction_small_tile(%input: tensor<32x32xf16>,
                                 %init: tensor<f16>) -> tensor<f16> {
  // CHECK: tiny_tile.load
  // CHECK: tiny_tile.reduce
  // CHECK: tensor.from_elements
  %result = linalg.generic {
    indexing_maps = [#map_input, #map_output],
    iterator_types = ["reduction", "reduction"]
  } ins(%input : tensor<32x32xf16>)
    outs(%init : tensor<f16>) {
  ^bb0(%in: f16, %out: f16):
    %sum = arith.addf %in, %out : f16
    linalg.yield %sum : f16
  } -> tensor<f16>
  return %result : tensor<f16>
}
