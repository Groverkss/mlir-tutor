//===- TinyTensorDialect.cpp - TinyTensor dialect implementation --*- C++ -*-===//
//
// Implements the TinyTensor dialect, including canonicalization patterns.
//
//===----------------------------------------------------------------------===//

#include "ch4-tensor-dsl/TinyTensorDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tiny_tensor;

//===----------------------------------------------------------------------===//
// TinyTensor dialect implementation
//===----------------------------------------------------------------------===//

#include "ch4-tensor-dsl/TinyTensorOpsDialect.cpp.inc"

void TinyTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ch4-tensor-dsl/TinyTensorOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {

/// Fold `to_ptr(from_ptr(x))` -> x
///
/// This pattern eliminates redundant conversions where a pointer is converted
/// to a tensor and immediately back to a pointer.
///
/// Example:
///   %tensor = tiny_tensor.from_ptr %ptr : !tiny.ptr -> tensor<64x128xf16>
///   %ptr2 = tiny_tensor.to_ptr %tensor : tensor<64x128xf16> -> !tiny.ptr
/// Becomes:
///   // %ptr2 is replaced with %ptr
struct FoldToPtrFromPtr : public OpRewritePattern<ToPtrOp> {
  using OpRewritePattern<ToPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToPtrOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input tensor comes from a from_ptr operation.
    auto fromPtrOp = op.getTensor().getDefiningOp<FromPtrOp>();
    if (!fromPtrOp)
      return failure();

    // Replace with the original pointer.
    rewriter.replaceOp(op, fromPtrOp.getPtr());
    return success();
  }
};

/// Fold `from_ptr(to_ptr(x))` -> x
///
/// This pattern eliminates redundant conversions where a tensor is converted
/// to a pointer and immediately back to a tensor. The tensor types must match.
///
/// Example:
///   %ptr = tiny_tensor.to_ptr %tensor : tensor<64x128xf16> -> !tiny.ptr
///   %tensor2 = tiny_tensor.from_ptr %ptr : !tiny.ptr -> tensor<64x128xf16>
/// Becomes:
///   // %tensor2 is replaced with %tensor
struct FoldFromPtrToPtr : public OpRewritePattern<FromPtrOp> {
  using OpRewritePattern<FromPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromPtrOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input pointer comes from a to_ptr operation.
    auto toPtrOp = op.getPtr().getDefiningOp<ToPtrOp>();
    if (!toPtrOp)
      return failure();

    // Check that the tensor types match.
    if (toPtrOp.getTensor().getType() != op.getResult().getType())
      return failure();

    // Replace with the original tensor.
    rewriter.replaceOp(op, toPtrOp.getTensor());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToPtrOp canonicalization
//===----------------------------------------------------------------------===//

void ToPtrOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<FoldToPtrFromPtr>(context);
}

//===----------------------------------------------------------------------===//
// FromPtrOp canonicalization
//===----------------------------------------------------------------------===//

void FromPtrOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<FoldFromPtrToPtr>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch4-tensor-dsl/TinyTensorOps.cpp.inc"
