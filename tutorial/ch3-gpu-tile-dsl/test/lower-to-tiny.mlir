// RUN: tutorial-opt --tiny-tile-layout-propagation --tiny-tile-to-tiny %s | FileCheck %s
// Test lowering of tiny_tile operations to tiny dialect.
// Layout: 16*4*8 = 512 elements, tile: 32x16 = 512 elements.

// CHECK-LABEL: func.func @lower_elementwise
func.func @lower_elementwise(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Load tiles with layout.
  // CHECK: gpu.thread_id x
  // CHECK: gpu.thread_id y
  // CHECK: tiny.load
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // CHECK: tiny.load
  %b = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Elementwise add becomes tiny.addf.
  // CHECK: tiny.addf
  %c = tiny_tile.elementwise add %a, %b
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
      -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Store becomes tiny.store with computed offset.
  // CHECK: tiny.store
  tiny_tile.store %c, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>, !tiny.ptr

  return
}
