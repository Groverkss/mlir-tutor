// RUN: tutorial-opt --split-input-file %s | FileCheck %s

func.func @empty() -> !list.list<i32> {
  %0 = list.empty : !list.list<i32>
  return %0 : !list.list<i32>
}

// CHECK-LABEL: func.func @empty
// CHECK: %[[LIST:.*]] = list.empty : <i32>
// CHECK: return %[[LIST]] : !list.list<i32>

// -----

func.func @append(%value: i32) -> !list.list<i32> {
  %empty = list.empty : !list.list<i32>
  %list = list.append %empty, %value : !list.list<i32>, i32 -> !list.list<i32>
  return %list : !list.list<i32>
}

// CHECK-LABEL: func.func @append
// CHECK: %[[EMPTY:.*]] = list.empty : <i32>
// CHECK: %[[LIST:.*]] = list.append %[[EMPTY]], %{{.*}} : <i32>, i32 -> <i32>
// CHECK: return %[[LIST]] : !list.list<i32>

// -----

func.func @concat_and_length(%lhs: !list.list<i32>, %rhs: !list.list<i32>) -> index {
  %list = list.concat %lhs, %rhs : !list.list<i32>, !list.list<i32> -> !list.list<i32>
  %length = list.length %list : !list.list<i32>
  return %length : index
}

// CHECK-LABEL: func.func @concat_and_length
// CHECK: %[[LIST:.*]] = list.concat %{{.*}}, %{{.*}} : <i32>, <i32> -> <i32>
// CHECK: %[[LENGTH:.*]] = list.length %[[LIST]] : <i32>
// CHECK: return %[[LENGTH]] : index
