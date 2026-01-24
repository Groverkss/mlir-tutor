// RUN: tutorial-opt %s | FileCheck %s
// Test parsing and printing of tiny_tile operations.

// CHECK-LABEL: func.func @test_elementwise
func.func @test_elementwise(%a: !tiny_tile.tile<64x128>, %b: !tiny_tile.tile<64x128>) -> !tiny_tile.tile<64x128> {
  // CHECK: tiny_tile.elementwise add
  %c = tiny_tile.elementwise add %a, %b : !tiny_tile.tile<64x128>, !tiny_tile.tile<64x128> -> !tiny_tile.tile<64x128>
  return %c : !tiny_tile.tile<64x128>
}

// CHECK-LABEL: func.func @test_elementwise_with_layout
// Layout: 16*4*8 = 512 elements, tile: 32x16 = 512 elements.
func.func @test_elementwise_with_layout(
    %a: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
    %b: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>)
    -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> {
  // CHECK: tiny_tile.elementwise mul
  // CHECK-SAME: #tiny_tile.layout<thread = [16, 4], vector_size = 8>
  %c = tiny_tile.elementwise mul %a, %b
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
      -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  return %c : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
}

// CHECK-LABEL: func.func @test_load_store
func.func @test_load_store(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // CHECK: tiny_tile.load
  %tile = tiny_tile.load %ptr, %row, %col stride %stride : !tiny.ptr -> !tiny_tile.tile<32x64>
  // CHECK: tiny_tile.store
  tiny_tile.store %tile, %ptr, %row, %col stride %stride : !tiny_tile.tile<32x64>, !tiny.ptr
  return
}

// CHECK-LABEL: func.func @test_reduce
func.func @test_reduce(%tile: !tiny_tile.tile<64x128>) -> f16 {
  // CHECK: tiny_tile.reduce
  %sum = tiny_tile.reduce %tile : !tiny_tile.tile<64x128> -> f16
  return %sum : f16
}

// CHECK-LABEL: func.func @test_layout_cast
func.func @test_layout_cast(%tile: !tiny_tile.tile<32x16>) -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> {
  // CHECK: tiny_tile.layout_cast
  %with_layout = tiny_tile.layout_cast %tile
      : !tiny_tile.tile<32x16> -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  return %with_layout : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
}
