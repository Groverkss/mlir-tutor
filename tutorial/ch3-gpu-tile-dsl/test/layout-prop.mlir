// RUN: tutorial-opt --tiny-tile-layout-propagation %s | FileCheck %s
// Test layout propagation pass.
// The pass should insert layout_cast operations to propagate layouts from
// load operations (source of truth) to other operations.

// Test 1: Elementwise with one operand missing layout.
// The layout should propagate from the load to the function argument via layout_cast.

// CHECK-LABEL: func.func @propagate_elementwise
func.func @propagate_elementwise(
    %arg: !tiny_tile.tile<32x16>,
    %ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Load has layout (source of truth).
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // CHECK: %[[CAST:.*]] = tiny_tile.layout_cast %arg0
  // CHECK-SAME: -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Result should also get layout via layout_cast.
  // CHECK: tiny_tile.elementwise add
  // CHECK: tiny_tile.layout_cast
  %b = tiny_tile.elementwise add %a, %arg
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16>
      -> !tiny_tile.tile<32x16>

  // CHECK: tiny_tile.store
  tiny_tile.store %b, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<32x16>, !tiny.ptr
  return
}

// Test 2: Chain of elementwise operations without layouts on results.
// Layouts should propagate forward through the chain.

// CHECK-LABEL: func.func @propagate_chain
func.func @propagate_chain(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Load tiles with layout.
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>
  %b = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>

  // Results have no layout - should get layout via layout_cast.
  // CHECK: tiny_tile.elementwise add
  // CHECK: tiny_tile.layout_cast
  // CHECK-SAME: -> !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>
  %c = tiny_tile.elementwise add %a, %b
       : !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>,
         !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>
      -> !tiny_tile.tile<16x16>

  // CHECK: tiny_tile.elementwise mul
  // CHECK: tiny_tile.layout_cast
  %d = tiny_tile.elementwise mul %c, %a
       : !tiny_tile.tile<16x16>,
         !tiny_tile.tile<16x16, #tiny_tile.layout<thread = [8, 4], vector_size = 8>>
      -> !tiny_tile.tile<16x16>

  tiny_tile.store %d, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<16x16>, !tiny.ptr
  return
}

// Test 3: Reduce operation - input without layout gets it from load.

// CHECK-LABEL: func.func @propagate_reduce
func.func @propagate_reduce(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) -> f16 {
  // Load with layout.
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Elementwise result without layout.
  // CHECK: tiny_tile.elementwise add
  // CHECK: tiny_tile.layout_cast
  %b = tiny_tile.elementwise add %a, %a
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
      -> !tiny_tile.tile<32x16>

  // Reduce takes the tile (layout_cast should be inserted before reduce).
  // CHECK: tiny_tile.reduce
  %sum = tiny_tile.reduce %b : !tiny_tile.tile<32x16> -> f16
  return %sum : f16
}

// Test 4: Splat provides layout (source of truth like load).

// CHECK-LABEL: func.func @propagate_splat
func.func @propagate_splat(%arg: !tiny_tile.tile<32x16>,
                           %ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Splat has layout (source of truth).
  // CHECK: tiny_tile.splat
  %tile = tiny_tile.splat 1.0 : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Argument needs layout_cast to match splat's layout.
  // CHECK: tiny_tile.layout_cast %arg0
  // CHECK-SAME: -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // CHECK: tiny_tile.elementwise add
  %sum = tiny_tile.elementwise add %tile, %arg
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16>
      -> !tiny_tile.tile<32x16>

  tiny_tile.store %sum, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<32x16>, !tiny.ptr
  return
}
