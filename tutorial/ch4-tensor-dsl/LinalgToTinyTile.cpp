//===- LinalgToTinyTile.cpp - Convert linalg to tiny_tile --------*- C++ -*-===//
//
// This pass converts linalg.generic operations to tiny_tile operations.
//
// The conversion follows a pattern similar to MLIR's linalg vectorizer,
// converting high-level tensor operations to tile-based operations.
//
//===----------------------------------------------------------------------===//

#include "ch4-tensor-dsl/TinyTensorPasses.h"
#include "ch4-tensor-dsl/TinyTensorDialect.h"
#include "ch1-cpu-vector-dsl/TinyDialect.h"
#include "ch3-gpu-tile-dsl/TinyTileDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tiny_tensor {

// Generate pass definitions.
#define GEN_PASS_DEF_LINALGTOTINYTILE
#include "ch4-tensor-dsl/TinyTensorPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Map an arith operation to the corresponding tiny_tile::ElementwiseKind.
/// Returns std::nullopt if the operation is not supported.
static std::optional<tiny_tile::ElementwiseKind>
mapArithOpToElementwiseKind(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<tiny_tile::ElementwiseKind>>(op)
      .Case<arith::AddFOp>([](auto) { return tiny_tile::ElementwiseKind::add; })
      .Case<arith::SubFOp>([](auto) { return tiny_tile::ElementwiseKind::sub; })
      .Case<arith::MulFOp>([](auto) { return tiny_tile::ElementwiseKind::mul; })
      .Case<arith::DivFOp>([](auto) { return tiny_tile::ElementwiseKind::div; })
      .Default([](auto) { return std::nullopt; });
}

/// Check if all iterator types are parallel.
static bool isAllParallel(linalg::GenericOp op) {
  for (auto iterType : op.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel)
      return false;
  }
  return true;
}

/// Check if all iterator types are reduction.
static bool isAllReduction(linalg::GenericOp op) {
  for (auto iterType : op.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::reduction)
      return false;
  }
  return true;
}

/// Check if a linalg.generic op is a simple elementwise operation.
///
/// Requirements:
/// - All iterator types are parallel (no reductions)
/// - All operands are 2D tensors of f16
/// - The body contains only supported arith operations
/// - Static tensor shapes
static bool isElementwiseGeneric(linalg::GenericOp op) {
  // Check all iterator types are parallel.
  if (!isAllParallel(op))
    return false;

  // Check that we have 2D tensors with f16 element type.
  for (auto operand : op.getInputs()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType || tensorType.getRank() != 2)
      return false;
    if (!tensorType.getElementType().isF16())
      return false;
    if (!tensorType.hasStaticShape())
      return false;
  }

  for (auto output : op.getOutputs()) {
    auto tensorType = dyn_cast<RankedTensorType>(output.getType());
    if (!tensorType || tensorType.getRank() != 2)
      return false;
    if (!tensorType.getElementType().isF16())
      return false;
    if (!tensorType.hasStaticShape())
      return false;
  }

  return true;
}

/// Check if a linalg.generic op is a full reduction.
///
/// Requirements:
/// - All iterator types are reduction
/// - Input is a 2D tensor of f16
/// - Output is a 0D tensor (scalar) of f16
/// - Static shapes
static bool isFullReduction(linalg::GenericOp op) {
  // Check all iterator types are reduction.
  if (!isAllReduction(op))
    return false;

  // Check that we have exactly one input.
  if (op.getInputs().size() != 1)
    return false;

  // Check input is 2D tensor of f16.
  auto inputType = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
  if (!inputType || inputType.getRank() != 2)
    return false;
  if (!inputType.getElementType().isF16())
    return false;
  if (!inputType.hasStaticShape())
    return false;

  // Check output is 0D tensor of f16.
  auto outputType = dyn_cast<RankedTensorType>(op.getOutputs().front().getType());
  if (!outputType || outputType.getRank() != 0)
    return false;
  if (!outputType.getElementType().isF16())
    return false;

  return true;
}

/// Get the tile type for a tensor type.
static tiny_tile::TileType getTileTypeForTensor(RankedTensorType tensorType,
                                                 MLIRContext *context) {
  int64_t height = tensorType.getDimSize(0);
  int64_t width = tensorType.getDimSize(1);
  return tiny_tile::TileType::get(context, height, width, /*layout=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Conversion pattern for elementwise linalg.generic
//===----------------------------------------------------------------------===//

/// Convert a simple elementwise linalg.generic to tiny_tile operations.
///
/// This pattern handles linalg.generic ops where:
/// - All operations are elementwise (parallel iterators only)
/// - All tensors are 2D with static shapes
/// - The body contains supported arith operations
///
/// The conversion:
/// 1. Converts input tensors to pointers (tiny_tensor.to_ptr)
/// 2. Loads tiles from pointers (tiny_tile.load)
/// 3. Converts scalar operations to tile operations (tiny_tile.elementwise)
/// 4. Stores result tile to pointer (tiny_tile.store)
/// 5. Converts output pointer back to tensor (tiny_tensor.from_ptr)
struct ElementwiseLinalgToTinyTile : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this is a supported elementwise operation.
    if (!isElementwiseGeneric(op))
      return failure();

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create constants for offsets.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Get the output tensor type to determine tile dimensions.
    auto outputTensorType = cast<RankedTensorType>(
        op.getOutputs().front().getType());
    int64_t width = outputTensorType.getDimSize(1);
    Value stride = rewriter.create<arith::ConstantIndexOp>(loc, width);

    // Create the tile type.
    auto tileType = getTileTypeForTensor(outputTensorType, context);

    // Convert input tensors to pointers and load tiles.
    SmallVector<Value> inputTiles;
    for (auto input : op.getInputs()) {
      // Convert tensor to pointer.
      Value ptr = rewriter.create<ToPtrOp>(loc,
          tiny::PtrType::get(context), input);

      // Load tile from pointer.
      Value tile = rewriter.create<tiny_tile::LoadOp>(loc, tileType, ptr,
                                                       c0, c0, stride);
      inputTiles.push_back(tile);
    }

    // Convert output tensor to pointer for storing result.
    Value outputPtr = rewriter.create<ToPtrOp>(loc,
        tiny::PtrType::get(context), op.getOutputs().front());

    // Process the body of the generic op.
    // We need to map block arguments to loaded tiles and convert operations.
    Block &body = op.getRegion().front();

    // Map block arguments to tiles.
    // Block args correspond to: inputs..., outputs...
    // We only use input tiles for computing; output is for storing.
    DenseMap<Value, Value> valueMapping;
    for (auto [idx, blockArg] : llvm::enumerate(body.getArguments())) {
      if (idx < inputTiles.size()) {
        valueMapping[blockArg] = inputTiles[idx];
      }
      // Output block args are not used in simple elementwise ops.
    }

    // Process operations in the body (excluding yield).
    Value resultTile;
    for (auto &bodyOp : body.without_terminator()) {
      auto kind = mapArithOpToElementwiseKind(&bodyOp);
      if (!kind) {
        return op.emitError() << "unsupported operation in linalg.generic body: "
                              << bodyOp.getName();
      }

      // Get operands from the mapping.
      Value lhs = valueMapping[bodyOp.getOperand(0)];
      Value rhs = valueMapping[bodyOp.getOperand(1)];

      // Create the elementwise operation.
      resultTile = rewriter.create<tiny_tile::ElementwiseOp>(
          loc, tileType, *kind, lhs, rhs);

      // Map the result for subsequent operations.
      valueMapping[bodyOp.getResult(0)] = resultTile;
    }

    // Store the result tile.
    rewriter.create<tiny_tile::StoreOp>(loc, resultTile, outputPtr,
                                         c0, c0, stride);

    // Convert output pointer back to tensor.
    Value resultTensor = rewriter.create<FromPtrOp>(loc, outputTensorType,
                                                     outputPtr);

    // Replace the linalg.generic with the result tensor.
    rewriter.replaceOp(op, resultTensor);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion pattern for reduction linalg.generic
//===----------------------------------------------------------------------===//

/// Convert a full reduction linalg.generic to tiny_tile.reduce.
///
/// This pattern handles linalg.generic ops where:
/// - All iterator types are reduction
/// - Input is a 2D tensor of f16
/// - Output is a scalar (0D tensor) of f16
///
/// The conversion:
/// 1. Converts input tensor to pointer (tiny_tensor.to_ptr)
/// 2. Loads tile from pointer (tiny_tile.load)
/// 3. Reduces tile to scalar (tiny_tile.reduce)
/// 4. Wraps scalar in 0D tensor (tensor.from_elements)
struct ReductionLinalgToTinyTile : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this is a supported reduction operation.
    if (!isFullReduction(op))
      return failure();

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create constants for offsets.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Get the input tensor type to determine tile dimensions.
    auto inputTensorType = cast<RankedTensorType>(
        op.getInputs().front().getType());
    int64_t width = inputTensorType.getDimSize(1);
    Value stride = rewriter.create<arith::ConstantIndexOp>(loc, width);

    // Create the tile type.
    auto tileType = getTileTypeForTensor(inputTensorType, context);

    // Convert input tensor to pointer and load tile.
    Value ptr = rewriter.create<ToPtrOp>(loc,
        tiny::PtrType::get(context), op.getInputs().front());
    Value tile = rewriter.create<tiny_tile::LoadOp>(loc, tileType, ptr,
                                                     c0, c0, stride);

    // Reduce the tile to a scalar.
    auto f16Type = rewriter.getF16Type();
    Value sum = rewriter.create<tiny_tile::ReduceOp>(loc, f16Type, tile);

    // Wrap the scalar in a 0D tensor.
    auto outputType = cast<RankedTensorType>(op.getOutputs().front().getType());
    Value result = rewriter.create<tensor::FromElementsOp>(loc, outputType,
                                                            ValueRange{sum});

    // Replace the linalg.generic with the result tensor.
    rewriter.replaceOp(op, result);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// LinalgToTinyTile pass implementation
//===----------------------------------------------------------------------===//

class LinalgToTinyTilePass : public impl::LinalgToTinyTileBase<LinalgToTinyTilePass> {
public:
  void runOnOperation() override {
    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ElementwiseLinalgToTinyTile>(&getContext());
    patterns.add<ReductionLinalgToTinyTile>(&getContext());

    // Apply patterns greedily to convert all matching linalg.generic ops.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny_tensor
