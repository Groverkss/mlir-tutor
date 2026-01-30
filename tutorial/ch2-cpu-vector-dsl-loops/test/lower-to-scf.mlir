// RUN: tutorial-opt --tiny-loop-to-scf %s | FileCheck %s
// Test lowering of tiny_loop.accumulate to scf.for.

// CHECK-LABEL: func.func @accumulate_lowering
func.func @accumulate_lowering(%n: index, %step: index, %init: index) -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (index)
  %result = tiny_loop.accumulate %n step(%step) init(%init) -> index {
  ^bb0(%i: index, %acc: index):
    // CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %[[IV]] : index
    %sum = arith.addi %acc, %i : index
    // CHECK: scf.yield %[[SUM]] : index
    tiny_loop.yield %sum : index
  }
  // CHECK: return %[[RESULT]]
  return %result : index
}

// CHECK-LABEL: func.func @nested_ops_preserved
func.func @nested_ops_preserved(%ptr: !tiny.ptr, %n: index, %step: index) {
  // Verify that operations inside the loop body are preserved.
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0]]
  tiny_loop.accumulate %n step(%step) {
  ^bb0(%i: index):
    // CHECK: tiny.load %{{.*}}, %[[IV]]
    %vec = tiny.load %ptr, %i : vector<4xf16>
    // CHECK: tiny.store
    tiny.store %vec, %ptr, %i : vector<4xf16>
  }
  return
}

// CHECK-LABEL: func.func @multiple_accumulators_lowering
func.func @multiple_accumulators_lowering(%n: index, %step: index,
                                          %init_a: i32, %init_b: f32) -> (i32, f32) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULTS:.*]]:2 = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, f32)
  %a, %b = tiny_loop.accumulate %n step(%step) init(%init_a, %init_b) -> (i32, f32) {
  ^bb0(%i: index, %acc_a: i32, %acc_b: f32):
    %one_i = arith.constant 1 : i32
    %one_f = arith.constant 1.0 : f32
    %next_a = arith.addi %acc_a, %one_i : i32
    %next_b = arith.addf %acc_b, %one_f : f32
    // CHECK: scf.yield %{{.*}}, %{{.*}} : i32, f32
    tiny_loop.yield %next_a, %next_b : i32, f32
  }
  // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1
  return %a, %b : i32, f32
}
