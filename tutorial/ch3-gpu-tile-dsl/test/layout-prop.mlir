// RUN: tutorial-opt --tiny-tile-layout-propagation %s | FileCheck %s
// Test layout propagation pass.

// Test 1: Function argument without layout, load provides layout.
// The layout_cast should be inserted to give the argument a layout.

// CHECK-LABEL: func.func @propagate_to_argument
func.func @propagate_to_argument(
    %arg: !tiny_tile.tile<32x16>,
    %ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Load has a layout, which is the source of truth.
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // CHECK: tiny_tile.layout_cast
  // CHECK-SAME: -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  // The argument needs a layout_cast to match the layout of %a.

  // CHECK: tiny_tile.elementwise add
  %b = tiny_tile.elementwise add %a, %arg
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16>
      -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  tiny_tile.store %b, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>, !tiny.ptr
  return
}

// Test 2: Chain of operations with layouts.

// CHECK-LABEL: func.func @propagate_chain
func.func @propagate_chain(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Load tiles with layout.
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>

  %b = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>

  // Elementwise with same layout as operands.
  // CHECK: tiny_tile.elementwise add
  // CHECK-SAME: #tiny_tile.layout<thread = [8, 4], vector_size = 8>
  %c = tiny_tile.elementwise add %a, %b
       : !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>,
         !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>
      -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>

  // Chain of operations.
  // CHECK: tiny_tile.elementwise mul
  // CHECK-SAME: #tiny_tile.layout<thread = [8, 4], vector_size = 8>
  %d = tiny_tile.elementwise mul %c, %a
       : !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>,
         !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>
      -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>

  tiny_tile.store %d, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>, !tiny.ptr
  return
}
