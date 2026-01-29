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

// CHECK-LABEL: func.func @lower_reduce
func.func @lower_reduce(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) -> f16 {
  // Load tile with layout.
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Reduce lowers to tiny.sum + vector.extract + gpu.all_reduce.
  // CHECK: tiny.sum
  // CHECK: vector.extract
  // CHECK: gpu.all_reduce add
  %sum = tiny_tile.reduce %a : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> -> f16
  return %sum : f16
}

// CHECK-LABEL: func.func @lower_splat
func.func @lower_splat(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) {
  // Splat lowers to tiny.constant with per-thread vector size.
  // CHECK: tiny.constant dense<1.000000e+00> : vector<8xf16>
  %tile = tiny_tile.splat 1.0 : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  tiny_tile.store %tile, %ptr, %row, %col stride %stride
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>, !tiny.ptr
  return
}
