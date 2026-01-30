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
// TinyTileToL2L1Op - Exercise for Students
//===----------------------------------------------------------------------===//

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
//  https://github.com/llvm/llvm-project/blob/a03c35fad7f1ac825da99f99ccf186e20c542eaa/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp#L3870-L3885
DiagnosedSilenceableFailure
TinyTileToL2L1Op::applyToOne(TransformRewriter &rewriter, Operation *target,
                             ApplyToEachResultList &results,
                             TransformState &state) {
  // Step 1: Check that the target implements TilingInterface.
  auto tilingInterface = dyn_cast<TilingInterface>(target);
  if (!tilingInterface) {
    return emitSilenceableError()
           << "target does not implement TilingInterface";
  }

  // Step 2: Convert L2 tile sizes from ArrayRef<int64_t> to OpFoldResult.
  SmallVector<OpFoldResult> l2TileSizes;
  for (int64_t size : getL2TileSizes()) {
    l2TileSizes.push_back(rewriter.getIndexAttr(size));
  }

  // Step 3: Create SCFTilingOptions for L2 tiling and tile.
  scf::SCFTilingOptions l2Options;
  l2Options.setTileSizes(l2TileSizes);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> l2Result =
      scf::tileUsingSCF(rewriter, tilingInterface, l2Options);
  if (failed(l2Result)) {
    return emitSilenceableError() << "failed to tile with L2 sizes";
  }

  rewriter.replaceOp(target, l2Result->replacements);

  // Step 4: Get the L2-tiled operation for L1 tiling.
  if (l2Result->tiledOps.empty()) {
    return emitSilenceableError() << "L2 tiling produced no tiled ops";
  }
  Operation *l2TiledOp = l2Result->tiledOps[0];

  // Step 5: Convert L1 tile sizes and tile again.
  SmallVector<OpFoldResult> l1TileSizes;
  for (int64_t size : getL1TileSizes()) {
    l1TileSizes.push_back(rewriter.getIndexAttr(size));
  }

  auto l1TilingInterface = dyn_cast<TilingInterface>(l2TiledOp);
  if (!l1TilingInterface) {
    return emitSilenceableError()
           << "L2-tiled op does not implement TilingInterface";
  }

  scf::SCFTilingOptions l1Options;
  l1Options.setTileSizes(l1TileSizes);

  rewriter.setInsertionPoint(l2TiledOp);
  FailureOr<scf::SCFTilingResult> l1Result =
      scf::tileUsingSCF(rewriter, l1TilingInterface, l1Options);
  if (failed(l1Result)) {
    return emitSilenceableError() << "failed to tile with L1 sizes";
  }

  rewriter.replaceOp(l2TiledOp, l1Result->replacements);

  // Step 6: Return the final tiled operation.
  if (l1Result->tiledOps.empty()) {
    return emitSilenceableError() << "L1 tiling produced no tiled ops";
  }
  results.push_back(l1Result->tiledOps[0]);

  return DiagnosedSilenceableFailure::success();
}

void TinyTileToL2L1Op::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Generated Op Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch4-linalg-transforms/TinyTransformOps.cpp.inc"
