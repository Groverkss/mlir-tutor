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
#include "mlir/Transforms/DialectConversion.h"

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

/// Lower tiny.subf to arith.subf.
struct SubFOpLowering : public OpRewritePattern<SubFOp> {
  using OpRewritePattern<SubFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubFOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SubFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Lower tiny.mulf to arith.mulf.
struct MulFOpLowering : public OpRewritePattern<MulFOp> {
  using OpRewritePattern<MulFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulFOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Lower tiny.divf to arith.divf.
struct DivFOpLowering : public OpRewritePattern<DivFOp> {
  using OpRewritePattern<DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DivFOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Lower tiny.addi to arith.addi.
struct AddIOpLowering : public OpRewritePattern<AddIOp> {
  using OpRewritePattern<AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddIOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Lower tiny.subi to arith.subi.
struct SubIOpLowering : public OpRewritePattern<SubIOp> {
  using OpRewritePattern<SubIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Lower tiny.muli to arith.muli.
struct MulIOpLowering : public OpRewritePattern<MulIOp> {
  using OpRewritePattern<MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Lower tiny.divi to arith.divsi (signed integer division).
struct DivIOpLowering : public OpRewritePattern<DivIOp> {
  using OpRewritePattern<DivIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DivIOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

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
    // Set up the conversion target.
    ConversionTarget target(getContext());

    // Mark arithmetic Tiny operations as illegal.
    target.addIllegalOp<ConstantOp, AddFOp, SubFOp, MulFOp, DivFOp, AddIOp,
                        SubIOp, MulIOp, DivIOp, SumOp>();

    // Mark arith and vector dialects as legal.
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect>();

    // Keep memory operations legal (they will be lowered by TinyToLLVM).
    target.addLegalOp<LoadOp, StoreOp>();

    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, AddFOpLowering, SubFOpLowering,
                 MulFOpLowering, DivFOpLowering, AddIOpLowering, SubIOpLowering,
                 MulIOpLowering, DivIOpLowering, SumOpLowering>(&getContext());

    // Apply the conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny
