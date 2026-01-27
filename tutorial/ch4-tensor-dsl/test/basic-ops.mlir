// RUN: tutorial-opt %s | FileCheck %s
// Test parsing and printing of tiny_tensor operations.

// CHECK-LABEL: func.func @test_to_ptr
func.func @test_to_ptr(%arg0: tensor<64x128xf16>) -> !tiny.ptr {
  // CHECK: tiny_tensor.to_ptr
  %ptr = tiny_tensor.to_ptr %arg0 : tensor<64x128xf16> -> !tiny.ptr
  return %ptr : !tiny.ptr
}

// CHECK-LABEL: func.func @test_from_ptr
func.func @test_from_ptr(%arg0: !tiny.ptr) -> tensor<64x128xf16> {
  // CHECK: tiny_tensor.from_ptr
  %tensor = tiny_tensor.from_ptr %arg0 : !tiny.ptr -> tensor<64x128xf16>
  return %tensor : tensor<64x128xf16>
}

// CHECK-LABEL: func.func @test_roundtrip
func.func @test_roundtrip(%arg0: tensor<32x64xf16>) -> tensor<32x64xf16> {
  // CHECK: tiny_tensor.to_ptr
  %ptr = tiny_tensor.to_ptr %arg0 : tensor<32x64xf16> -> !tiny.ptr
  // CHECK: tiny_tensor.from_ptr
  %tensor = tiny_tensor.from_ptr %ptr : !tiny.ptr -> tensor<32x64xf16>
  return %tensor : tensor<32x64xf16>
}
