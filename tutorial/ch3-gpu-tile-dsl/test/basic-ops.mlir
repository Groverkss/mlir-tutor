// RUN: tutorial-opt --split-input-file %s | FileCheck %s
// Test parsing and printing of tiny_tile operations.
// All tile types must have a layout.

#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>
#L2 = #tiny_tile.layout<thread = [8, 8], vector_size = 8>

// CHECK-LABEL: func.func @test_elementwise
func.func @test_elementwise(
    %a: !tiny_tile.tile<32x16, #L>,
    %b: !tiny_tile.tile<32x16, #L>) -> !tiny_tile.tile<32x16, #L> {
  // CHECK: tiny_tile.elementwise add
  %c = tiny_tile.elementwise add %a, %b
       : !tiny_tile.tile<32x16, #L>, !tiny_tile.tile<32x16, #L>
      -> !tiny_tile.tile<32x16, #L>
  return %c : !tiny_tile.tile<32x16, #L>
}

// -----

#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>

// CHECK-LABEL: func.func @test_all_elementwise_kinds
func.func @test_all_elementwise_kinds(
    %a: !tiny_tile.tile<32x16, #L>,
    %b: !tiny_tile.tile<32x16, #L>) -> !tiny_tile.tile<32x16, #L> {
  // CHECK: tiny_tile.elementwise add
  %add = tiny_tile.elementwise add %a, %b
       : !tiny_tile.tile<32x16, #L>, !tiny_tile.tile<32x16, #L>
      -> !tiny_tile.tile<32x16, #L>
  // CHECK: tiny_tile.elementwise sub
  %sub = tiny_tile.elementwise sub %a, %b
       : !tiny_tile.tile<32x16, #L>, !tiny_tile.tile<32x16, #L>
      -> !tiny_tile.tile<32x16, #L>
  // CHECK: tiny_tile.elementwise mul
  %mul = tiny_tile.elementwise mul %a, %b
       : !tiny_tile.tile<32x16, #L>, !tiny_tile.tile<32x16, #L>
      -> !tiny_tile.tile<32x16, #L>
  // CHECK: tiny_tile.elementwise div
  %div = tiny_tile.elementwise div %a, %b
       : !tiny_tile.tile<32x16, #L>, !tiny_tile.tile<32x16, #L>
      -> !tiny_tile.tile<32x16, #L>
  return %div : !tiny_tile.tile<32x16, #L>
}

// -----

#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>

// CHECK-LABEL: func.func @test_load_store
func.func @test_load_store(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // CHECK: tiny_tile.load
  %tile = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #L>
  // CHECK: tiny_tile.store
  tiny_tile.store %tile, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<32x16, #L>, !tiny.ptr
  return
}

// -----

#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>

// CHECK-LABEL: func.func @test_sum
func.func @test_sum(%tile: !tiny_tile.tile<32x16, #L>) -> vector<1xf16> {
  // CHECK: tiny_tile.sum
  %sum = tiny_tile.sum %tile : !tiny_tile.tile<32x16, #L> -> vector<1xf16>
  return %sum : vector<1xf16>
}

// -----

#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>

// CHECK-LABEL: func.func @test_splat
func.func @test_splat() -> !tiny_tile.tile<32x16, #L> {
  // CHECK: tiny_tile.splat 1.000000e+00
  %tile = tiny_tile.splat 1.0 : !tiny_tile.tile<32x16, #L>
  return %tile : !tiny_tile.tile<32x16, #L>
}

// -----

#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>

// CHECK-LABEL: func.func @test_splat_zero
func.func @test_splat_zero() -> !tiny_tile.tile<32x16, #L> {
  // CHECK: tiny_tile.splat 0.000000e+00
  %tile = tiny_tile.splat 0.0 : !tiny_tile.tile<32x16, #L>
  return %tile : !tiny_tile.tile<32x16, #L>
}
