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

// CHECK-LABEL: func.func @lower_elementwise_sub
func.func @lower_elementwise_sub(
    %a: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
    %b: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>)
    -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> {
  // CHECK: tiny.subf
  %c = tiny_tile.elementwise sub %a, %b
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
      -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  return %c : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
}

// CHECK-LABEL: func.func @lower_elementwise_mul
func.func @lower_elementwise_mul(
    %a: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
    %b: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>)
    -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> {
  // CHECK: tiny.mulf
  %c = tiny_tile.elementwise mul %a, %b
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
      -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  return %c : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
}

// CHECK-LABEL: func.func @lower_elementwise_div
func.func @lower_elementwise_div(
    %a: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
    %b: !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>)
    -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> {
  // CHECK: tiny.divf
  %c = tiny_tile.elementwise div %a, %b
       : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>,
         !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
      -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
  return %c : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
}

// CHECK-LABEL: func.func @lower_sum
func.func @lower_sum(%ptr: !tiny.ptr, %row: index, %col: index, %stride: index) -> vector<1xf16> {
  // Load tile with layout.
  %a = tiny_tile.load %ptr, %row, %col stride %stride
       : !tiny.ptr -> !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>

  // Sum lowers to tiny.sum + vector.extract + gpu.subgroup_reduce + vector.broadcast.
  // CHECK: tiny.sum
  // CHECK: vector.extract
  // CHECK: gpu.subgroup_reduce add
  // CHECK: vector.broadcast
  %sum = tiny_tile.sum %a : !tiny_tile.tile<32x16, #tiny_tile.layout<thread = [16, 4], vector_size = 8>> -> vector<1xf16>
  return %sum : vector<1xf16>
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
