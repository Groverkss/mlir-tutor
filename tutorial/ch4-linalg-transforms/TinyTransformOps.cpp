//===- TinyTransformOps.cpp - Transform ops for ch4 tutorial --------------===//
//
// This file implements custom transform dialect operations for the tutorial.
//
//===----------------------------------------------------------------------===//

#include "ch4-linalg-transforms/TinyTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// TinyLoopUnrollFullOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
TinyLoopUnrollFullOp::applyToOne(TransformRewriter &rewriter, Operation *target,
                             ApplyToEachResultList &results,
                             TransformState &state) {
  // Step 1: Check that the target is an scf.for operation.
  auto forOp = dyn_cast<scf::ForOp>(target);
  if (!forOp) {
    return emitSilenceableError()
           << "expected scf.for, but got " << target->getName();
  }

  // Step 2: Attempt to fully unroll the loop.
  // loopUnrollFull requires constant bounds and will fail otherwise.
  if (failed(loopUnrollFull(forOp))) {
    return emitSilenceableError()
           << "failed to fully unroll loop (bounds may not be constant)";
  }

  // Step 3: Return success.
  return DiagnosedSilenceableFailure::success();
}

void TinyLoopUnrollFullOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // We only read the target handle (we don't consume or produce handles).
  onlyReadsHandle(getTargetMutable(), effects);
  // We modify the payload IR (the loop gets unrolled).
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// TinyTileToL2L1Op - Exercise 3
//===----------------------------------------------------------------------===//

// TODO: Implement TinyTileToL2L1Op::applyToOne
//
// Steps:
//  1. Cast target to TilingInterface using dyn_cast<TilingInterface>(target)
//  2. Create SCFTilingOptions for L2 tiling:
//     ```cpp
//     scf::SCFTilingOptions l2Options;
//     l2Options.setTileSizes(...);  // Use getL2TileSizes()
//     ```
//  3. Call scf::tileUsingSCF(rewriter, tilingInterface, l2Options)
//  4. Replace the original op: rewriter.replaceOp(target, l2Result->replacements)
//  5. Repeat for L1 tiling on l2Result->tiledOps[0]
//  6. Return the final tiled op via results.push_back()
//
//  Reference:
//  https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp
//
// DiagnosedSilenceableFailure
// TinyTileToL2L1Op::applyToOne(TransformRewriter &rewriter, Operation *target,
//                              ApplyToEachResultList &results,
//                              TransformState &state) {
//   // Your implementation here
//   return DiagnosedSilenceableFailure::success();
// }
//
// Hint: getEffects() declares what the transform does to handles and payload:
//
// void TinyTileToL2L1Op::getEffects(
//     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
//   // We only read the target handle (don't consume it)
//   onlyReadsHandle(getTargetMutable(), effects);
//   // We produce a new handle for the tiled operation
//   producesHandle(getOperation()->getOpResults(), effects);
//   // We modify the payload IR (tiling changes the IR)
//   modifiesPayload(effects);
// }

//===----------------------------------------------------------------------===//
// Generated Op Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch4-linalg-transforms/TinyTransformOps.cpp.inc"
