//===- TinyLoopPasses.td - TinyLoop dialect passes ---------*- tablegen -*-===//
//
// Defines passes for the TinyLoop dialect (Chapter 2).
//
//===----------------------------------------------------------------------===//

#ifndef TINY_LOOP_PASSES
#define TINY_LOOP_PASSES

include "mlir/Pass/PassBase.td"

def TinyLoopToSCF : Pass<"tiny-loop-to-scf"> {
  let summary = "Lower TinyLoop operations to SCF dialect.";
  let description = [{
    This pass lowers TinyLoop dialect operations to equivalent operations
    in the SCF (Structured Control Flow) dialect.

    The lowering includes:
    - `tiny_loop.accumulate` -> `scf.for` (with lower bound 0)
    - `tiny_loop.yield` -> `scf.yield`

    Example:
    ```mlir
    %result = tiny_loop.accumulate %n step(%step) init(%init) -> index {
    ^bb0(%i: index, %acc: index):
      %next = arith.addi %acc, %i : index
      tiny_loop.yield %next : index
    }
    // Becomes:
    %c0 = arith.constant 0 : index
    %result = scf.for %i = %c0 to %n step %step iter_args(%acc = %init) -> index {
      %next = arith.addi %acc, %i : index
      scf.yield %next : index
    }
    ```

    Note: Run this pass before --tiny-to-arith if you have Tiny arithmetic ops
    in the loop body.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // TINY_LOOP_PASSES
