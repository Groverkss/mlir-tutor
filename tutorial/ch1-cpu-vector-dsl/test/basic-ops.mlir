// RUN: tutorial-opt --split-input-file %s | FileCheck %s
// Test basic parsing and printing of Tiny dialect operations.

// CHECK-LABEL: func.func @test_vector_constant
func.func @test_vector_constant() -> vector<4xf16> {
  // CHECK: tiny.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf16>
  %0 = tiny.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_index_constant
func.func @test_index_constant() -> index {
  // CHECK: tiny.constant 42 : index
  %0 = tiny.constant 42 : index
  return %0 : index
}

// -----

// CHECK-LABEL: func.func @test_vector_ops
func.func @test_vector_ops(%a: vector<4xf16>, %b: vector<4xf16>) -> vector<4xf16> {
  // CHECK: tiny.addf
  %0 = tiny.addf %a, %b : vector<4xf16>
  // CHECK: tiny.subf
  %1 = tiny.subf %0, %b : vector<4xf16>
  // CHECK: tiny.mulf
  %2 = tiny.mulf %1, %a : vector<4xf16>
  // CHECK: tiny.divf
  %3 = tiny.divf %2, %b : vector<4xf16>
  return %3 : vector<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_index_ops
func.func @test_index_ops(%a: index, %b: index) -> index {
  // CHECK: tiny.addi
  %0 = tiny.addi %a, %b
  // CHECK: tiny.subi
  %1 = tiny.subi %0, %b
  // CHECK: tiny.muli
  %2 = tiny.muli %1, %a
  // CHECK: tiny.divi
  %3 = tiny.divi %2, %b
  return %3 : index
}

// -----

// CHECK-LABEL: func.func @test_sum
func.func @test_sum(%a: vector<4xf16>) -> vector<1xf16> {
  // CHECK: tiny.sum %{{.*}} : vector<4xf16> -> vector<1xf16>
  %0 = tiny.sum %a : vector<4xf16> -> vector<1xf16>
  return %0 : vector<1xf16>
}

// -----

// CHECK-LABEL: func.func @test_memory_ops
func.func @test_memory_ops(%ptr: !tiny.ptr, %offset: index, %vec: vector<4xf16>) {
  // CHECK: tiny.load %{{.*}}, %{{.*}} : vector<4xf16>
  %0 = tiny.load %ptr, %offset : vector<4xf16>
  // CHECK: tiny.store %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf16>
  tiny.store %vec, %ptr, %offset : vector<4xf16>
  return
}
