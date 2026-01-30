// RUN: tutorial-opt --transform-interpreter --split-input-file %s | FileCheck %s
// Test the reference implementation: transform.tiny.loop_unroll_full

// CHECK-LABEL: func.func @unroll_simple_loop
func.func @unroll_simple_loop() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c0_idx = arith.constant 0 : index
  %c4_idx = arith.constant 4 : index
  %c1_idx = arith.constant 1 : index

  // This loop should be fully unrolled (4 iterations).
  // CHECK-NOT: scf.for
  // CHECK: arith.addi
  // CHECK: arith.addi
  // CHECK: arith.addi
  // CHECK: arith.addi
  %result = scf.for %i = %c0_idx to %c4_idx step %c1_idx iter_args(%acc = %c0) -> i32 {
    %new_acc = arith.addi %acc, %c1 : i32
    scf.yield %new_acc : i32
  }

  return %result : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.tiny.loop_unroll_full %loop : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @unroll_with_iter_args
func.func @unroll_with_iter_args(%init: f32) -> f32 {
  %c0_idx = arith.constant 0 : index
  %c3_idx = arith.constant 3 : index
  %c1_idx = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32

  // Loop with 3 iterations should unroll completely.
  // CHECK-NOT: scf.for
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  %result = scf.for %i = %c0_idx to %c3_idx step %c1_idx iter_args(%acc = %init) -> f32 {
    %new_acc = arith.mulf %acc, %c2 : f32
    scf.yield %new_acc : f32
  }

  return %result : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.tiny.loop_unroll_full %loop : !transform.any_op
    transform.yield
  }
}
