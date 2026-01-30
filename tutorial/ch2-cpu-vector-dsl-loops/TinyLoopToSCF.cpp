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

#include "ch2-cpu-vector-dsl-loops/TinyLoopDialect.h"
#include "ch2-cpu-vector-dsl-loops/TinyLoopPasses.h"

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

/// Lower tiny_loop.accumulate to scf.for. scf.for op definition:
/// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop
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
///
/// Note: that this implementation is not the easiest, but is very useful
/// to understand the IR structure. When in doubt, look at the solution
/// on `main`, and take motivation from there and implement it.
///
/// You can implement this using a OpRewritePattern on the AccumulateOp.
/// The recommended way of doing this is:
///
/// 1. Create a new scf::ForOp with the lower bound, upper bound, step, and
///    iter arguments. This will create a scf::ForOp with an empty body,
///    but the iter arguments and everything will be setup.
///
/// 2. Set the insertion point inside the loop body. You can do this
///    using rewriter.setInsertionPointToStart .
///
/// 3. Setup an IRMapping:
///    https://github.com/llvm/llvm-project/blob/ac47d8c227ce1c2413ad70d95d6cf7f7f09b6943/mlir/include/mlir/IR/IRMapping.h#L26
///    An IRMapping is nothing but just a map, which stores a mapping from
///    an old value to a new value. We are maintaining this mapping so
///    that we can clone the operations from the accumulate body to
///    the scf.for body, but as we do it, the values used by operations
///    need to be mapped to the newly cloned ones.
///
/// 4. Add a mapping from old block arguments to the new block arguments.
///    You can do this by running a loop and mapping each old body argument
///    to the new body argument using IRMapping::map
///
/// 5. For each operation in the accumulate body except the last operation
///    (terminator), clone it using rewriter.clone . Note that the clone method
///    takes an IRMapping object and will automatically add the result to the
///    mapping, so you just need to run clone on every operation.
///
/// 6. Create a yield operation, with the operands as mapped inputs from
///    the old yield operation. You can use IRMapping::lookup for this.
///    Hint: mlir::Block has a getTerminator function.
///
///    Only create this yield if there are any loop carried variables. There
///    is a quirk with the scf::ForOp builder that it creates a scf.yield
///    automatically if there are no loop carried variables.
///
/// 7. Replace the accumulate operation with the new scf.for operation.
struct AccumulateOpLowering : public OpRewritePattern<AccumulateOp> {
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Create the lower bound constant (always 0).
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto forOp = scf::ForOp::create(rewriter, loc, zero, op.getBound(),
                                    op.getStep(), op.getInitArgs());

    // Clone each operation to the forOp's body.
    rewriter.setInsertionPointToStart(forOp.getBody());

    IRMapping mapping;
    for (int64_t i = 0; i < op.getBody()->getNumArguments(); ++i) {
      mapping.map(op.getBody()->getArgument(i),
                  forOp.getBody()->getArgument(i));
    }

    for (Operation &bodyOp : op.getBody()->getOperations()) {
      if (isa<tiny_loop::YieldOp>(bodyOp)) {
        continue;
      }
      rewriter.clone(bodyOp, mapping);
    }

    SmallVector<Value> newOperands;
    for (Value val : op.getBody()->getTerminator()->getOperands()) {
      newOperands.push_back(mapping.lookup(val));
    }

    if (!newOperands.empty()) {
      scf::YieldOp::create(rewriter, loc, newOperands);
    }

    // Replace the accumulate op results with the scf.for results.
    rewriter.replaceOp(op, forOp.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// TinyLoopToSCF pass implementation
//===----------------------------------------------------------------------===//

class TinyLoopToSCFPass : public impl::TinyLoopToSCFBase<TinyLoopToSCFPass> {
public:
  void runOnOperation() override {
    // We implement this pass similar to ch1's TinyToArith pass. We
    // will use the greedy pattern driver to write this. Feel free
    // to copy the rewriter boilter plate from ch1's TinyToArith pass.

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
