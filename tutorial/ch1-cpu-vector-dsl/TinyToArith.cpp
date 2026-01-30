//===- TinyToArith.cpp - Lower Tiny arithmetic ops to arith -----*- C++ -*-===//
//
// This pass lowers Tiny dialect arithmetic operations to the arith dialect.
// Memory operations (load/store) are NOT converted by this pass.
//
//===----------------------------------------------------------------------===//

#include "ch1-cpu-vector-dsl/TinyPasses.h"
#include "ch1-cpu-vector-dsl/TinyDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tiny {

// Generate pass definitions.
#define GEN_PASS_DEF_TINYTOARITH
#include "ch1-cpu-vector-dsl/TinyPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Lower tiny.constant to arith.constant.
struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(),
                                                   op.getValue());
    return success();
  }
};

/// Lower tiny.addf to arith.addf.
struct AddFOpLowering : public OpRewritePattern<AddFOp> {
  using OpRewritePattern<AddFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddFOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

// TODO: Implement SubFOpLowering
// Lower tiny.subf to arith.subf
// Hint: Follow the same pattern as AddFOpLowering

// TODO: Implement MulFOpLowering
// Lower tiny.mulf to arith.mulf
// Hint: Follow the same pattern as AddFOpLowering

// TODO: Implement DivFOpLowering
// Lower tiny.divf to arith.divf
// Hint: Follow the same pattern as AddFOpLowering

/// Lower tiny.addi to arith.addi.
struct AddIOpLowering : public OpRewritePattern<AddIOp> {
  using OpRewritePattern<AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddIOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

// TODO: Implement SubIOpLowering
// Lower tiny.subi to arith.subi
// Hint: Follow the same pattern as AddIOpLowering

// TODO: Implement MulIOpLowering
// Lower tiny.muli to arith.muli
// Hint: Follow the same pattern as AddIOpLowering

// TODO: Implement DivIOpLowering
// Lower tiny.divi to arith.divsi (signed integer division)
// Hint: Follow the same pattern as AddIOpLowering

/// Lower tiny.sum to vector.reduction<add> + vector.broadcast.
struct SumOpLowering : public OpRewritePattern<SumOp> {
  using OpRewritePattern<SumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SumOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();

    // vector.reduction<add> returns a scalar f16.
    Value scalarSum = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::ADD, input);

    // Broadcast the scalar to vector<1xf16>.
    VectorType resultType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resultType, scalarSum);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// TinyToArith pass implementation
//===----------------------------------------------------------------------===//

class TinyToArithPass : public impl::TinyToArithBase<TinyToArithPass> {
public:
  void runOnOperation() override {
    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, AddFOpLowering, SumOpLowering, AddIOpLowering>(&getContext());
    // TODO: Add SubFOpLowering, MulFOpLowering, DivFOpLowering, SubIOpLowering, MulIOpLowering, DivIOpLowering

    // Apply patterns greedily to lower all matching operations.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny
