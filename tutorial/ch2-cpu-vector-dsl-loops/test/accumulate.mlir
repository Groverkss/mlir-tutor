// RUN: tutorial-opt --split-input-file %s | FileCheck %s
// Test tiny_loop.accumulate with init args (loop-carried values).

// CHECK-LABEL: func.func @sum_loop
func.func @sum_loop(%n: index, %step: index, %init: index) -> index {
  // CHECK: tiny_loop.accumulate %{{.*}} step(%{{.*}}) init(%{{.*}}) -> index {
  // CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
  %result = tiny_loop.accumulate %n step(%step) init(%init) -> index {
  ^bb0(%i: index, %acc: index):
    %next = arith.addi %acc, %i : index
    // CHECK: tiny_loop.yield %{{.*}} : index
    tiny_loop.yield %next : index
  }
  return %result : index
}

// -----

// CHECK-LABEL: func.func @vector_accumulation
func.func @vector_accumulation(%ptr: !tiny.ptr, %n: index,
                               %init: vector<4xf16>) -> vector<4xf16> {
  %c4 = arith.constant 4 : index
  // CHECK: tiny_loop.accumulate %{{.*}} step(%{{.*}}) init(%{{.*}}) -> vector<4xf16>
  %result = tiny_loop.accumulate %n step(%c4) init(%init) -> vector<4xf16> {
  ^bb0(%i: index, %acc: vector<4xf16>):
    // Load a vector and accumulate.
    %vec = tiny.load %ptr, %i : vector<4xf16>
    %sum = arith.addf %acc, %vec : vector<4xf16>
    tiny_loop.yield %sum : vector<4xf16>
  }
  return %result : vector<4xf16>
}

// -----

// CHECK-LABEL: func.func @multiple_accumulators
func.func @multiple_accumulators(%n: index, %step: index,
                                 %init_a: i32, %init_b: f32) -> (i32, f32) {
  // CHECK: tiny_loop.accumulate %{{.*}} step(%{{.*}}) init(%{{.*}}, %{{.*}}) -> (i32, f32)
  %a, %b = tiny_loop.accumulate %n step(%step) init(%init_a, %init_b) -> (i32, f32) {
  ^bb0(%i: index, %acc_a: i32, %acc_b: f32):
    %one_i = arith.constant 1 : i32
    %one_f = arith.constant 1.0 : f32
    %next_a = arith.addi %acc_a, %one_i : i32
    %next_b = arith.addf %acc_b, %one_f : f32
    // CHECK: tiny_loop.yield %{{.*}}, %{{.*}} : i32, f32
    tiny_loop.yield %next_a, %next_b : i32, f32
  }
  return %a, %b : i32, f32
}
