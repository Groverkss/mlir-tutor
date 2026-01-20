// RUN: tutorial-opt --tiny-loop-to-scf --tiny-to-arith --tiny-to-llvm %s | FileCheck %s
// Test full lowering pipeline: tiny_loop ops + tiny ops to LLVM.

// CHECK-LABEL: func.func @vector_sum_pipeline
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr, %[[N:.*]]: index)
func.func @vector_sum_pipeline(%ptr: !tiny.ptr, %n: index) -> vector<4xf16> {
  %c4 = tiny.constant 4 : index
  %zero = tiny.constant dense<0.0> : vector<4xf16>

  // The full pipeline should produce:
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[N]]
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: arith.addf
  // CHECK: scf.yield
  %result = tiny_loop.accumulate %n step(%c4) init(%zero) -> vector<4xf16> {
  ^bb0(%i: index, %acc: vector<4xf16>):
    %vec = tiny.load %ptr, %i : vector<4xf16>
    %sum = tiny.addf %acc, %vec : vector<4xf16>
    tiny_loop.yield %sum : vector<4xf16>
  }

  return %result : vector<4xf16>
}

// CHECK-LABEL: func.func @index_sum_pipeline
func.func @index_sum_pipeline(%n: index) -> index {
  %c1 = tiny.constant 1 : index
  %zero = tiny.constant 0 : index

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: scf.for
  // CHECK: arith.addi
  // CHECK: scf.yield
  %result = tiny_loop.accumulate %n step(%c1) init(%zero) -> index {
  ^bb0(%i: index, %acc: index):
    %sum = tiny.addi %acc, %i
    tiny_loop.yield %sum : index
  }

  return %result : index
}
