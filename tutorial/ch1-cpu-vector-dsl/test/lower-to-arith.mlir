// RUN: tutorial-opt --tiny-to-arith %s | FileCheck %s
// Test lowering of Tiny dialect arithmetic operations to arith dialect.
// Note: Memory operations (load/store) are NOT lowered by this pass.

// CHECK-LABEL: func.func @test_vector_constant_lowering
func.func @test_vector_constant_lowering() -> vector<4xf16> {
  // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf16>
  // CHECK-NOT: tiny.constant
  %0 = tiny.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf16>
  return %0 : vector<4xf16>
}

// CHECK-LABEL: func.func @test_index_constant_lowering
func.func @test_index_constant_lowering() -> index {
  // CHECK: arith.constant 42 : index
  // CHECK-NOT: tiny.constant
  %0 = tiny.constant 42 : index
  return %0 : index
}

// CHECK-LABEL: func.func @test_vector_ops_lowering
func.func @test_vector_ops_lowering(%a: vector<4xf16>, %b: vector<4xf16>) -> vector<4xf16> {
  // CHECK: arith.addf
  // CHECK-NOT: tiny.addf
  %0 = tiny.addf %a, %b : vector<4xf16>
  // CHECK: arith.subf
  // CHECK-NOT: tiny.subf
  %1 = tiny.subf %0, %b : vector<4xf16>
  // CHECK: arith.mulf
  // CHECK-NOT: tiny.mulf
  %2 = tiny.mulf %1, %a : vector<4xf16>
  // CHECK: arith.divf
  // CHECK-NOT: tiny.divf
  %3 = tiny.divf %2, %b : vector<4xf16>
  return %3 : vector<4xf16>
}

// CHECK-LABEL: func.func @test_index_ops_lowering
func.func @test_index_ops_lowering(%a: index, %b: index) -> index {
  // CHECK: arith.addi
  // CHECK-NOT: tiny.addi
  %0 = tiny.addi %a, %b
  // CHECK: arith.subi
  // CHECK-NOT: tiny.subi
  %1 = tiny.subi %0, %b
  // CHECK: arith.muli
  // CHECK-NOT: tiny.muli
  %2 = tiny.muli %1, %a
  // CHECK: arith.divsi
  // CHECK-NOT: tiny.divi
  %3 = tiny.divi %2, %b
  return %3 : index
}

// CHECK-LABEL: func.func @test_sum_lowering
func.func @test_sum_lowering(%a: vector<4xf16>) -> vector<1xf16> {
  // CHECK: vector.reduction <add>, %{{.*}} : vector<4xf16> into f16
  // CHECK: vector.broadcast %{{.*}} : f16 to vector<1xf16>
  // CHECK-NOT: tiny.sum
  %0 = tiny.sum %a : vector<4xf16> -> vector<1xf16>
  return %0 : vector<1xf16>
}

// CHECK-LABEL: func.func @test_memory_ops_preserved
// Memory operations should NOT be lowered by tiny-to-arith pass.
// CHECK-SAME: (%[[PTR:.*]]: !tiny.ptr, %[[OFFSET:.*]]: index)
func.func @test_memory_ops_preserved(%ptr: !tiny.ptr, %offset: index) -> vector<4xf16> {
  // CHECK: tiny.load %[[PTR]], %[[OFFSET]] : vector<4xf16>
  %0 = tiny.load %ptr, %offset : vector<4xf16>
  // CHECK: tiny.store
  tiny.store %0, %ptr, %offset : vector<4xf16>
  return %0 : vector<4xf16>
}
