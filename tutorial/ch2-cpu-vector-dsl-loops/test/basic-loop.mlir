// RUN: tutorial-opt --split-input-file %s | FileCheck %s
// Test basic parsing and printing of tiny_loop.accumulate.

// CHECK-LABEL: func.func @side_effect_loop
func.func @side_effect_loop(%ptr: !tiny.ptr, %n: index, %step: index) {
  // CHECK: tiny_loop.accumulate %{{.*}} step(%{{.*}}) {
  // CHECK-NEXT: ^bb0(%{{.*}}: index):
  tiny_loop.accumulate %n step(%step) {
  ^bb0(%i: index):
    // CHECK: tiny.load
    %vec = tiny.load %ptr, %i : vector<4xf16>
    // CHECK: tiny.store
    tiny.store %vec, %ptr, %i : vector<4xf16>
  }
  return
}

// -----

// CHECK-LABEL: func.func @loop_with_constant_step
func.func @loop_with_constant_step(%ptr: !tiny.ptr, %n: index) {
  %c4 = arith.constant 4 : index
  // CHECK: tiny_loop.accumulate %{{.*}} step(%{{.*}}) {
  tiny_loop.accumulate %n step(%c4) {
  ^bb0(%i: index):
    %vec = tiny.load %ptr, %i : vector<4xf16>
    tiny.store %vec, %ptr, %i : vector<4xf16>
  }
  return
}
