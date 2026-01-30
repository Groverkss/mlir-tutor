//===- TinyTransformOps.td - Transform ops for ch4 tutorial ---------------===//
//
// This file defines custom transform dialect operations for the tutorial.
// These operations demonstrate how to extend the transform dialect with
// custom transformations.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TRANSFORM_OPS_TD
#define TINY_TRANSFORM_OPS_TD

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TinyLoopUnrollFullOp
//===----------------------------------------------------------------------===//

def TinyLoopUnrollFullOp : TransformDialectOp<"tiny.loop_unroll_full",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Fully unrolls a loop with constant bounds";
  let description = [{
    This operation fully unrolls an `scf.for` loop that has constant bounds.
    The loop body is replicated for each iteration and the loop is eliminated.

    This is a reference implementation for the tutorial. Students can study
    this to understand how to implement their own transform operations.

    Example:
    ```mlir
    %loop = transform.structured.match ops{["scf.for"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.tiny.loop_unroll_full %loop : !transform.any_op
    ```

    The operation fails if:
    - The target is not an `scf.for` operation
    - The loop bounds are not constant
    - The unrolling fails for any other reason
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// TinyTileToL2L1Op - Exercise 3
//===----------------------------------------------------------------------===//

// TODO: Define TinyTileToL2L1Op
//
// This operation should perform two-level cache tiling in one step.
// First tiles with L2 sizes, then tiles the result with L1 sizes.
//
// Example usage:
// ```mlir
// %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
//     : (!transform.any_op) -> !transform.any_op
// %tiled = transform.tiny.tile_to_l2_l1 %matmul
//     l2 [64, 64, 32] l1 [16, 16, 8]
//     : (!transform.any_op) -> !transform.any_op
// ```
//
// Hints:
// 1. Use TransformDialectOp<"tiny.tile_to_l2_l1", [...]> as the base
// 2. Include these traits: TransformOpInterface, TransformEachOpTrait,
//    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
//    ReportTrackingListenerFailuresOpTrait
// 3. Arguments: target (TransformHandleTypeInterface), l2_tile_sizes (DenseI64ArrayAttr),
//    l1_tile_sizes (DenseI64ArrayAttr)
// 4. Results: tiled_op (TransformHandleTypeInterface)
// 5. Look at TinyLoopUnrollFullOp above for reference

#endif // TINY_TRANSFORM_OPS_TD
