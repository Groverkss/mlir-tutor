// RUN: tutorial-opt %s -canonicalize | FileCheck %s
// Test canonicalization patterns for tiny_tensor operations.

// CHECK-LABEL: func.func @fold_to_ptr_from_ptr
// CHECK-SAME: (%[[ARG:.*]]: !tiny.ptr)
func.func @fold_to_ptr_from_ptr(%arg0: !tiny.ptr) -> !tiny.ptr {
  %tensor = tiny_tensor.from_ptr %arg0 : !tiny.ptr -> tensor<64x128xf16>
  %ptr = tiny_tensor.to_ptr %tensor : tensor<64x128xf16> -> !tiny.ptr
  // CHECK: return %[[ARG]] : !tiny.ptr
  return %ptr : !tiny.ptr
}

// CHECK-LABEL: func.func @fold_from_ptr_to_ptr
// CHECK-SAME: (%[[ARG:.*]]: tensor<64x128xf16>)
func.func @fold_from_ptr_to_ptr(%arg0: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %ptr = tiny_tensor.to_ptr %arg0 : tensor<64x128xf16> -> !tiny.ptr
  %tensor = tiny_tensor.from_ptr %ptr : !tiny.ptr -> tensor<64x128xf16>
  // CHECK: return %[[ARG]] : tensor<64x128xf16>
  return %tensor : tensor<64x128xf16>
}

// CHECK-LABEL: func.func @no_fold_different_types
// CHECK-SAME: (%[[ARG:.*]]: tensor<64x128xf16>)
func.func @no_fold_different_types(%arg0: tensor<64x128xf16>) -> tensor<32x64xf16> {
  // This should NOT fold because the tensor types are different.
  %ptr = tiny_tensor.to_ptr %arg0 : tensor<64x128xf16> -> !tiny.ptr
  // CHECK: tiny_tensor.to_ptr
  // CHECK: tiny_tensor.from_ptr
  %tensor = tiny_tensor.from_ptr %ptr : !tiny.ptr -> tensor<32x64xf16>
  return %tensor : tensor<32x64xf16>
}

// CHECK-LABEL: func.func @fold_chain
// CHECK-SAME: (%[[ARG:.*]]: !tiny.ptr)
func.func @fold_chain(%arg0: !tiny.ptr) -> !tiny.ptr {
  // Multiple conversions should all fold away.
  %t1 = tiny_tensor.from_ptr %arg0 : !tiny.ptr -> tensor<64x128xf16>
  %p1 = tiny_tensor.to_ptr %t1 : tensor<64x128xf16> -> !tiny.ptr
  %t2 = tiny_tensor.from_ptr %p1 : !tiny.ptr -> tensor<64x128xf16>
  %p2 = tiny_tensor.to_ptr %t2 : tensor<64x128xf16> -> !tiny.ptr
  // CHECK: return %[[ARG]] : !tiny.ptr
  return %p2 : !tiny.ptr
}
