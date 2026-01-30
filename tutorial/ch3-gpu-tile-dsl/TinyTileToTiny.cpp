//===- TinyTileToTiny.cpp - Convert tiny_tile to tiny dialect ----*- C++
//-*-===//
//
// This pass converts tiny_tile operations to per-thread tiny vector operations.
//
// The conversion uses gpu.thread_id to get the thread position and computes
// per-thread offsets based on the layout attribute.
//
//===----------------------------------------------------------------------===//

#include "ch1-cpu-vector-dsl/TinyDialect.h"
#include "ch3-gpu-tile-dsl/TinyTileDialect.h"
#include "ch3-gpu-tile-dsl/TinyTilePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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

    // Convert TileType to VectorType (layout is always present).
    addConversion([](TileType tileType) -> Type {
      return tileType.getPerThreadVectorType();
    });
  }
};

//===----------------------------------------------------------------------===//
// Generic interface-based conversion pattern
//===----------------------------------------------------------------------===//

/// A generic pattern that converts any operation implementing
/// TinyTileLoweringOpInterface by calling its convertToSIMT method.
struct TinyTileLoweringPattern
    : public OpInterfaceConversionPattern<TinyTileLoweringOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(TinyTileLoweringOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return op.convertToSIMT(rewriter, operands);
  }
};

//===----------------------------------------------------------------------===//
// TinyTileToTiny pass implementation
//===----------------------------------------------------------------------===//

class TinyTileToTinyPass : public impl::TinyTileToTinyBase<TinyTileToTinyPass> {
public:
  void runOnOperation() override {
    TinyTileToTinyTypeConverter typeConverter;

    ConversionTarget target(getContext());

    // Mark tiny_tile operations as illegal.
    target.addIllegalDialect<TinyTileDialect>();

    // Mark target dialects as legal.
    target.addLegalDialect<tiny::TinyDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    // Mark SCF ops as dynamically legal (legal if types are converted).
    target.addDynamicallyLegalDialect<scf::SCFDialect>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());

    // Single pattern handles all ops via the interface.
    patterns.add<TinyTileLoweringPattern>(typeConverter, &getContext());

    // Add SCF structural type conversion patterns.
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);

    // Apply the conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny_tile
