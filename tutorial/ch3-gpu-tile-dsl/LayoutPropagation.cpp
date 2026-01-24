//===- LayoutPropagation.cpp - Layout propagation pass ----------*- C++ -*-===//
//
// This pass propagates layout attributes through tiny_tile operations by
// inserting layout_cast operations where needed.
//
// The propagation uses the TinyTileLayoutInterface to determine how layouts
// flow through operations. When an operation requires a layout that doesn't
// match its operand, a layout_cast is inserted.
//
//===----------------------------------------------------------------------===//

#include "ch3-gpu-tile-dsl/TinyTilePasses.h"
#include "ch3-gpu-tile-dsl/TinyTileDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tiny_tile {

// Generate pass definitions.
#define GEN_PASS_DEF_TINYTILELAYOUTPROPAGATION
#include "ch3-gpu-tile-dsl/TinyTilePasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Get the layout from a tile type, or nullptr if no layout or not a tile.
static LayoutAttr getLayout(Type type) {
  if (auto tileType = dyn_cast<TileType>(type))
    return tileType.getLayout();
  return nullptr;
}

/// Get the layout from a value's type.
static LayoutAttr getLayout(Value value) { return getLayout(value.getType()); }

/// Create a new tile type with the given layout.
static TileType getTileTypeWithLayout(TileType tileType, LayoutAttr layout) {
  return TileType::get(tileType.getContext(), tileType.getHeight(),
                       tileType.getWidth(), layout);
}

//===----------------------------------------------------------------------===//
// LayoutPropagation pass implementation
//===----------------------------------------------------------------------===//

class TinyTileLayoutPropagationPass
    : public impl::TinyTileLayoutPropagationBase<TinyTileLayoutPropagationPass> {
public:
  void runOnOperation() override {
    // First pass: collect all layouts from load operations (sources of truth).
    // Then propagate layouts forward through operations, inserting layout_cast
    // where needed.

    getOperation()->walk([&](Operation *op) {
      // Skip layout_cast operations.
      if (isa<LayoutCastOp>(op))
        return;

      auto layoutInterface = dyn_cast<TinyTileLayoutInterface>(op);
      if (!layoutInterface)
        return;

      // Collect operand layouts.
      SmallVector<LayoutAttr> operandLayouts;
      for (Value operand : op->getOperands())
        operandLayouts.push_back(getLayout(operand));

      // For each result, infer what layout it should have.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        auto resultType = dyn_cast<TileType>(op->getResult(i).getType());
        if (!resultType)
          continue;

        // If result already has a layout, nothing to do.
        if (resultType.getLayout())
          continue;

        // Try to infer layout from operands.
        LayoutAttr inferredLayout =
            layoutInterface.inferResultLayout(i, operandLayouts);
        if (!inferredLayout)
          continue;

        // Insert a layout_cast after this operation to add the layout.
        OpBuilder builder(op->getContext());
        builder.setInsertionPointAfter(op);

        auto newResultType = getTileTypeWithLayout(resultType, inferredLayout);
        auto castOp = LayoutCastOp::create(builder, op->getLoc(), newResultType,
                                           op->getResult(i));

        // Replace all uses of the old result with the cast result.
        op->getResult(i).replaceAllUsesExcept(castOp.getResult(), castOp);
      }
    });

    // Second pass: for operations that need operands with layouts, insert
    // layout_cast before them.
    getOperation()->walk([&](Operation *op) {
      // Skip layout_cast operations.
      if (isa<LayoutCastOp>(op))
        return;

      auto layoutInterface = dyn_cast<TinyTileLayoutInterface>(op);
      if (!layoutInterface)
        return;

      // Collect result layouts.
      SmallVector<LayoutAttr> resultLayouts;
      for (Type resultType : op->getResultTypes())
        resultLayouts.push_back(getLayout(resultType));

      // For each operand, check if it needs a layout.
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Value operand = op->getOperand(i);
        auto operandType = dyn_cast<TileType>(operand.getType());
        if (!operandType)
          continue;

        // Get the operand's current layout.
        LayoutAttr currentLayout = operandType.getLayout();

        // Infer what layout this operand should have.
        LayoutAttr requiredLayout =
            layoutInterface.inferOperandLayout(i, resultLayouts);

        // If layouts match or no layout required, nothing to do.
        if (currentLayout == requiredLayout || !requiredLayout)
          continue;

        // Insert a layout_cast to convert the operand to the required layout.
        OpBuilder builder(op->getContext());
        builder.setInsertionPoint(op);

        auto newOperandType = getTileTypeWithLayout(operandType, requiredLayout);
        auto castOp =
            LayoutCastOp::create(builder, op->getLoc(), newOperandType, operand);

        op->setOperand(i, castOp.getResult());
      }
    });

    // Verify all tile results have layouts (except for layout_cast inputs).
    bool hasError = false;
    getOperation()->walk([&](Operation *op) {
      // Skip checking layout_cast - it's allowed to have unspecified input.
      if (isa<LayoutCastOp>(op))
        return;

      for (auto result : op->getResults()) {
        if (auto tileType = dyn_cast<TileType>(result.getType())) {
          // Check if all users of this result have layouts.
          for (Operation *user : result.getUsers()) {
            // If user is a layout_cast, that's fine.
            if (isa<LayoutCastOp>(user))
              continue;

            // Otherwise, if this result has no layout and is used by a
            // non-layout_cast op, that's an error.
            if (!tileType.getLayout()) {
              op->emitError() << "tile result has no layout after propagation";
              hasError = true;
              break;
            }
          }
        }
      }
    });

    if (hasError)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny_tile
