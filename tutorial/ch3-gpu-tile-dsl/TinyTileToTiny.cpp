//===- TinyTileToTiny.cpp - Convert tiny_tile to tiny dialect ----*- C++ -*-===//
//
// This pass converts tiny_tile operations to per-thread tiny vector operations.
//
// The conversion uses gpu.thread_id to get the thread position and computes
// per-thread offsets based on the layout attribute.
//
//===----------------------------------------------------------------------===//

#include "ch3-gpu-tile-dsl/TinyTilePasses.h"
#include "ch3-gpu-tile-dsl/TinyTileDialect.h"
#include "ch1-cpu-vector-dsl/TinyDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tiny_tile {

// Generate pass definitions.
#define GEN_PASS_DEF_TINYTILETOTINY
#include "ch3-gpu-tile-dsl/TinyTilePasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

/// Type converter that converts tile types to vector types based on layout.
class TinyTileToTinyTypeConverter : public TypeConverter {
public:
  TinyTileToTinyTypeConverter() {
    // Identity conversion for all types (fallback).
    addConversion([](Type type) { return type; });

    // Convert TileType with layout to VectorType.
    addConversion([](TileType tileType) -> Type {
      if (!tileType.hasLayout()) {
        // Cannot convert tiles without layout.
        return nullptr;
      }
      return tileType.getPerThreadVectorType();
    });
  }
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Compute the per-thread offset within a tile based on thread IDs and layout.
///
/// Given layout {thread=[rows, cols], vector_size=vs}:
///   linear_tid = tid_y * rows + tid_x
///   thread_offset = linear_tid * vs
static Value computeThreadOffset(OpBuilder &builder, Location loc,
                                 LayoutAttr layout) {
  // Get thread IDs.
  Value tidX = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value tidY = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::y);

  // Get layout parameters.
  ArrayRef<int64_t> threadDims = layout.getThread();
  int64_t threadRows = threadDims[0];
  int64_t vectorSize = layout.getVectorSize();

  // Compute linear thread ID: tid_y * thread[0] + tid_x
  Value threadRowsConst =
      arith::ConstantOp::create(builder, loc, builder.getIndexAttr(threadRows));
  Value linearTid =
      arith::AddIOp::create(builder, loc, tidX,
          arith::MulIOp::create(builder, loc, tidY, threadRowsConst));

  // Compute thread offset: linear_tid * vector_size
  Value vectorSizeConst =
      arith::ConstantOp::create(builder, loc, builder.getIndexAttr(vectorSize));
  return arith::MulIOp::create(builder, loc, linearTid, vectorSizeConst);
}

/// Compute the base offset in memory: row * stride + col.
static Value computeBaseOffset(OpBuilder &builder, Location loc, Value row,
                               Value col, Value stride) {
  Value rowOffset = arith::MulIOp::create(builder, loc, row, stride);
  return arith::AddIOp::create(builder, loc, rowOffset, col);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Convert tiny_tile.elementwise to tiny.addf/subf/mulf/divf.
struct ElementwiseOpLowering : public OpConversionPattern<ElementwiseOp> {
  using OpConversionPattern<ElementwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Create the appropriate tiny operation based on kind.
    switch (op.getKind()) {
    case ElementwiseKind::add:
      rewriter.replaceOpWithNewOp<tiny::AddFOp>(op, lhs, rhs);
      break;
    case ElementwiseKind::sub:
      rewriter.replaceOpWithNewOp<tiny::SubFOp>(op, lhs, rhs);
      break;
    case ElementwiseKind::mul:
      rewriter.replaceOpWithNewOp<tiny::MulFOp>(op, lhs, rhs);
      break;
    case ElementwiseKind::div:
      rewriter.replaceOpWithNewOp<tiny::DivFOp>(op, lhs, rhs);
      break;
    }

    return success();
  }
};

/// Convert tiny_tile.load to tiny.load with per-thread offset.
struct LoadOpLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the tile type and its layout.
    auto tileType = cast<TileType>(op.getResult().getType());
    LayoutAttr layout = tileType.getLayout();
    if (!layout) {
      return op.emitError() << "load requires tile with layout";
    }

    // Compute base offset: row * stride + col.
    Value baseOffset = computeBaseOffset(rewriter, loc, adaptor.getRow(),
                                         adaptor.getCol(), adaptor.getStride());

    // Compute per-thread offset.
    Value threadOffset = computeThreadOffset(rewriter, loc, layout);

    // Final offset = base + thread_offset.
    Value finalOffset =
        arith::AddIOp::create(rewriter, loc, baseOffset, threadOffset);

    // Create tiny.load with the computed offset.
    VectorType vectorType = tileType.getPerThreadVectorType();
    rewriter.replaceOpWithNewOp<tiny::LoadOp>(op, vectorType, adaptor.getPtr(),
                                              finalOffset);

    return success();
  }
};

/// Convert tiny_tile.store to tiny.store with per-thread offset.
struct StoreOpLowering : public OpConversionPattern<StoreOp> {
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the tile type and its layout.
    auto tileType = cast<TileType>(op.getValue().getType());
    LayoutAttr layout = tileType.getLayout();
    if (!layout) {
      return op.emitError() << "store requires tile with layout";
    }

    // Compute base offset: row * stride + col.
    Value baseOffset = computeBaseOffset(rewriter, loc, adaptor.getRow(),
                                         adaptor.getCol(), adaptor.getStride());

    // Compute per-thread offset.
    Value threadOffset = computeThreadOffset(rewriter, loc, layout);

    // Final offset = base + thread_offset.
    Value finalOffset =
        arith::AddIOp::create(rewriter, loc, baseOffset, threadOffset);

    // Create tiny.store with the computed offset.
    rewriter.replaceOpWithNewOp<tiny::StoreOp>(op, adaptor.getValue(),
                                               adaptor.getPtr(), finalOffset);

    return success();
  }
};

/// Convert tiny_tile.reduce to tiny.sum + gpu.all_reduce.
struct ReduceOpLowering : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern<ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // First, reduce the per-thread vector to a scalar using tiny.sum.
    VectorType vec1Type = VectorType::get({1}, rewriter.getF16Type());
    Value localSum =
        tiny::SumOp::create(rewriter, loc, vec1Type, adaptor.getInput());

    // Extract the scalar from vector<1xf16>.
    Value scalar =
        vector::ExtractOp::create(rewriter, loc, localSum, ArrayRef<int64_t>{0});

    // Perform cross-thread reduction using gpu.all_reduce.
    // Set up properties for the all_reduce operation.
    gpu::AllReduceOp::Properties props;
    props.op = gpu::AllReduceOperationAttr::get(rewriter.getContext(),
                                                gpu::AllReduceOperation::ADD);
    Value globalSum = gpu::AllReduceOp::create(rewriter, loc, scalar.getType(),
                                               ValueRange{scalar}, props);

    rewriter.replaceOp(op, globalSum);
    return success();
  }
};

/// Convert tiny_tile.layout_cast - just forwards the value since layout is
/// only a type annotation. After conversion, both input and output have the
/// same vector type.
struct LayoutCastOpLowering : public OpConversionPattern<LayoutCastOp> {
  using OpConversionPattern<LayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LayoutCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Simply forward the input - layout_cast is just a type annotation.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TinyTileToTiny pass implementation
//===----------------------------------------------------------------------===//

class TinyTileToTinyPass : public impl::TinyTileToTinyBase<TinyTileToTinyPass> {
public:
  void runOnOperation() override {
    // Set up the type converter.
    TinyTileToTinyTypeConverter typeConverter;

    // Set up the conversion target.
    ConversionTarget target(getContext());

    // Mark tiny_tile operations as illegal.
    target.addIllegalDialect<TinyTileDialect>();

    // Mark target dialects as legal.
    target.addLegalDialect<tiny::TinyDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ElementwiseOpLowering, LoadOpLowering, StoreOpLowering,
                 ReduceOpLowering, LayoutCastOpLowering>(typeConverter,
                                                         &getContext());

    // Apply the conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny_tile
