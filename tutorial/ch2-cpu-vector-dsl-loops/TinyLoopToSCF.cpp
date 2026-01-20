//===- TinyLoopToSCF.cpp - Lower TinyLoop ops to SCF ------------*- C++ -*-===//
//
// This pass lowers TinyLoop dialect operations to the SCF dialect.
//
// This demonstrates:
// - Converting operations with regions
// - Cloning and remapping regions
// - Block argument conversion
//
//===----------------------------------------------------------------------===//

#include "ch2-cpu-vector-dsl-loops/TinyLoopPasses.h"
#include "ch2-cpu-vector-dsl-loops/TinyLoopDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tiny_loop {

// Generate pass definitions.
#define GEN_PASS_DEF_TINYLOOPTOSCF
#include "ch2-cpu-vector-dsl-loops/TinyLoopPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Lower tiny_loop.accumulate to scf.for.
///
/// The transformation is:
///
/// tiny_loop.accumulate %N step(%step) init(%init) -> T {
/// ^bb0(%i: index, %acc: T):
///   %next = ...
///   tiny_loop.yield %next : T
/// }
///
/// Becomes:
///
/// %c0 = arith.constant 0 : index
/// scf.for %i = %c0 to %N step %step iter_args(%acc = %init) -> T {
///   %next = ...
///   scf.yield %next : T
/// }
struct AccumulateOpLowering : public OpRewritePattern<AccumulateOp> {
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create the lower bound constant (always 0).
    Value lowerBound =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));

    // Get the upper bound and step from the accumulate op.
    Value upperBound = op.getBound();
    Value step = op.getStep();

    // Get the body block from the tiny_loop.accumulate.
    Block *tinyBody = op.getBody();
    auto tinyYield = cast<YieldOp>(tinyBody->getTerminator());

    // Create the scf.for operation with a body builder that creates a
    // yield terminator. We'll then clone the rest of the body.
    auto scfForOp = scf::ForOp::create(
        rewriter, loc, lowerBound, upperBound, step, op.getInitArgs(),
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          // Create a mapping from old block arguments to new block arguments.
          IRMapping mapping;
          mapping.map(tinyBody->getArgument(0), iv);
          for (auto [oldArg, newArg] :
               llvm::zip(tinyBody->getArguments().drop_front(1), iterArgs)) {
            mapping.map(oldArg, newArg);
          }

          // Clone all operations from the tiny_loop.accumulate body,
          // except for the terminator.
          for (Operation &bodyOp : tinyBody->without_terminator()) {
            builder.clone(bodyOp, mapping);
          }

          // Create the scf.yield with the mapped yield operands.
          SmallVector<Value> yieldOperands;
          for (Value operand : tinyYield.getResults()) {
            yieldOperands.push_back(mapping.lookupOrDefault(operand));
          }
          scf::YieldOp::create(builder, loc, yieldOperands);
        });

    // Replace the accumulate op results with the scf.for results.
    rewriter.replaceOp(op, scfForOp.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// TinyLoopToSCF pass implementation
//===----------------------------------------------------------------------===//

class TinyLoopToSCFPass : public impl::TinyLoopToSCFBase<TinyLoopToSCFPass> {
public:
  void runOnOperation() override {
    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<AccumulateOpLowering>(&getContext());

    // Apply patterns greedily to lower all matching operations.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny_loop
