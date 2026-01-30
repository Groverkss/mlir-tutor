// RUN: tutorial-opt --split-input-file --tiny-to-llvm %s | FileCheck %s
// Test lowering of Tiny dialect memory operations to LLVM dialect.

// CHECK-LABEL: func.func @test_load_lowering
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[OFFSET:.*]]: index)
func.func @test_load_lowering(%ptr: !tiny.ptr, %offset: index) -> vector<4xf16> {
  // CHECK: %[[OFFSET_I64:.*]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[OFFSET_I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
  // CHECK: %[[VEC:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> vector<4xf16>
  // CHECK-NOT: tiny.load
  %0 = tiny.load %ptr, %offset : vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_store_lowering
// CHECK-SAME: (%[[VEC:.*]]: vector<4xf16>, %[[PTR:.*]]: !llvm.ptr, %[[OFFSET:.*]]: index)
func.func @test_store_lowering(%vec: vector<4xf16>, %ptr: !tiny.ptr, %offset: index) {
  // CHECK: %[[OFFSET_I64:.*]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[OFFSET_I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
  // CHECK: llvm.store %[[VEC]], %[[GEP]] : vector<4xf16>, !llvm.ptr
  // CHECK-NOT: tiny.store
  tiny.store %vec, %ptr, %offset : vector<4xf16>
  return
}

// -----

// CHECK-LABEL: func.func @test_load_store_pipeline
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[OFFSET:.*]]: index)
func.func @test_load_store_pipeline(%ptr: !tiny.ptr, %offset: index) -> vector<4xf16> {
  // Load a vector, then store it at offset + 4.
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  %vec = tiny.load %ptr, %offset : vector<4xf16>

  %c4 = arith.constant 4 : index
  %next_offset = arith.addi %offset, %c4 : index

  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  tiny.store %vec, %ptr, %next_offset : vector<4xf16>

  return %vec : vector<4xf16>
}

// -----

// CHECK-LABEL: func.func @test_type_conversion
// Verify that the !tiny.ptr type is converted to !llvm.ptr in function signatures.
// CHECK-SAME: (%[[ARG:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @test_type_conversion(%ptr: !tiny.ptr) -> !tiny.ptr {
  // CHECK: return %[[ARG]] : !llvm.ptr
  return %ptr : !tiny.ptr
}
