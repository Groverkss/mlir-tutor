//===- TinyTileDialect.cpp - TinyTile dialect implementation -----*- C++ -*-===//
//
// Implements the TinyTile dialect, including verifiers and interface methods.
//
//===----------------------------------------------------------------------===//

#include "ch3-gpu-tile-dsl/TinyTileDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::tiny_tile;

//===----------------------------------------------------------------------===//
// TinyTile dialect implementation
//===----------------------------------------------------------------------===//

#include "ch3-gpu-tile-dsl/TinyTileOpsDialect.cpp.inc"

void TinyTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ch3-gpu-tile-dsl/TinyTileOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "ch3-gpu-tile-dsl/TinyTileTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "ch3-gpu-tile-dsl/TinyTileAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LayoutAttr verification
//===----------------------------------------------------------------------===//

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> thread, int64_t vectorSize) {
  // Thread dimensions must be exactly 2.
  if (thread.size() != 2) {
    return emitError() << "thread must have exactly 2 dimensions, got "
                       << thread.size();
  }

  // Each thread dimension must be a power of 2.
  for (auto [idx, dim] : llvm::enumerate(thread)) {
    if (!llvm::isPowerOf2_64(dim)) {
      return emitError() << "thread dimension " << idx << " must be a power of 2, got "
                         << dim;
    }
  }

  // Vector size must be a power of 2.
  if (!llvm::isPowerOf2_64(vectorSize)) {
    return emitError() << "vector_size must be a power of 2, got " << vectorSize;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TileType verification
//===----------------------------------------------------------------------===//

LogicalResult TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                               int64_t height, int64_t width,
                               LayoutAttr layout) {
  // Height and width must be powers of 2.
  if (!llvm::isPowerOf2_64(height)) {
    return emitError() << "tile height must be a power of 2, got " << height;
  }
  if (!llvm::isPowerOf2_64(width)) {
    return emitError() << "tile width must be a power of 2, got " << width;
  }

  // If layout is present, verify tile elements match layout elements.
  if (layout) {
    int64_t tileElements = height * width;
    int64_t layoutElements = layout.getNumElements();
    if (tileElements != layoutElements) {
      return emitError() << "tile has " << tileElements << " elements but layout covers "
                         << layoutElements << " elements";
    }
  }

  return success();
}

VectorType TileType::getPerThreadVectorType() const {
  if (!hasLayout())
    return nullptr;

  auto layout = getLayout();
  auto elementType = Float16Type::get(getContext());
  return VectorType::get({layout.getVectorSize()}, elementType);
}

//===----------------------------------------------------------------------===//
// ElementwiseOp verification and interface methods
//===----------------------------------------------------------------------===//

LogicalResult ElementwiseOp::verify() {
  auto lhsType = cast<TileType>(getLhs().getType());
  auto rhsType = cast<TileType>(getRhs().getType());
  auto resultType = cast<TileType>(getResult().getType());

  // All operands and result must have the same dimensions.
  if (lhsType.getHeight() != rhsType.getHeight() ||
      lhsType.getWidth() != rhsType.getWidth()) {
    return emitOpError() << "operands must have the same dimensions, got "
                         << lhsType.getHeight() << "x" << lhsType.getWidth()
                         << " vs " << rhsType.getHeight() << "x"
                         << rhsType.getWidth();
  }

  if (lhsType.getHeight() != resultType.getHeight() ||
      lhsType.getWidth() != resultType.getWidth()) {
    return emitOpError() << "result must have the same dimensions as operands, got "
                         << resultType.getHeight() << "x" << resultType.getWidth()
                         << " vs " << lhsType.getHeight() << "x"
                         << lhsType.getWidth();
  }

  return success();
}

LayoutAttr ElementwiseOp::inferResultLayout(
    unsigned resultIndex, ArrayRef<LayoutAttr> operandLayouts) {
  // Result layout is the same as operand layouts (they must match).
  for (auto layout : operandLayouts)
    if (layout)
      return layout;
  return nullptr;
}

LayoutAttr ElementwiseOp::inferOperandLayout(
    unsigned operandIndex, ArrayRef<LayoutAttr> resultLayouts) {
  // Operands need the same layout as results.
  for (auto layout : resultLayouts)
    if (layout)
      return layout;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ReduceOp interface methods
//===----------------------------------------------------------------------===//

LayoutAttr ReduceOp::inferResultLayout(unsigned resultIndex,
                                       ArrayRef<LayoutAttr> operandLayouts) {
  // Result is a scalar, no layout.
  return nullptr;
}

LayoutAttr ReduceOp::inferOperandLayout(unsigned operandIndex,
                                        ArrayRef<LayoutAttr> resultLayouts) {
  // Cannot infer input layout from scalar output.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// LoadOp interface methods
//===----------------------------------------------------------------------===//

LayoutAttr LoadOp::inferResultLayout(unsigned resultIndex,
                                     ArrayRef<LayoutAttr> operandLayouts) {
  // Result layout comes from the result type (source of truth for loads).
  auto resultType = cast<TileType>(getResult().getType());
  return resultType.getLayout();
}

LayoutAttr LoadOp::inferOperandLayout(unsigned operandIndex,
                                      ArrayRef<LayoutAttr> resultLayouts) {
  // Load has no tile operands, nothing to infer.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// StoreOp interface methods
//===----------------------------------------------------------------------===//

LayoutAttr StoreOp::inferResultLayout(unsigned resultIndex,
                                      ArrayRef<LayoutAttr> operandLayouts) {
  // Store has no results.
  return nullptr;
}

LayoutAttr StoreOp::inferOperandLayout(unsigned operandIndex,
                                       ArrayRef<LayoutAttr> resultLayouts) {
  // The value operand (index 0) should use the layout from the value type.
  if (operandIndex == 0) {
    auto valueType = cast<TileType>(getValue().getType());
    return valueType.getLayout();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// LayoutCastOp verification
//===----------------------------------------------------------------------===//

LogicalResult LayoutCastOp::verify() {
  auto inputType = cast<TileType>(getInput().getType());
  auto resultType = cast<TileType>(getResult().getType());

  // Input and output must have the same dimensions.
  if (inputType.getHeight() != resultType.getHeight() ||
      inputType.getWidth() != resultType.getWidth()) {
    return emitOpError() << "input and output tiles must have the same "
                         << "dimensions, got " << inputType.getHeight() << "x"
                         << inputType.getWidth() << " vs "
                         << resultType.getHeight() << "x"
                         << resultType.getWidth();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// TileType custom parser/printer
//===----------------------------------------------------------------------===//
//
// We need custom parsing because the MLIR lexer tokenizes "64x128" as a
// dimension list token (not as separate integers). The built-in assemblyFormat
// cannot handle this, so we use parser.parseDimensionList().
//

/// Parse a tile type: `<` HxW (`,` layout)? `>`
/// Example: <64x128> or <64x128, #tiny_tile.layout<thread=[16,4], vector_size=8>>
Type TileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse "HxW" as a dimension list. MLIR's lexer treats "64x128" as a single
  // dimension list token, so we must use parseDimensionList.
  SmallVector<int64_t, 2> dims;
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false,
                                /*withTrailingX=*/false))
    return Type();

  // We expect exactly 2 dimensions for a 2D tile.
  if (dims.size() != 2) {
    parser.emitError(parser.getCurrentLocation())
        << "expected 2 dimensions for tile, got " << dims.size();
    return Type();
  }

  // Parse optional layout attribute.
  LayoutAttr layout;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(layout))
      return Type();
  }

  if (parser.parseGreater())
    return Type();

  return TileType::get(parser.getContext(), dims[0], dims[1], layout);
}

/// Print a tile type: `<` HxW (`,` layout)? `>`
void TileType::print(AsmPrinter &printer) const {
  printer << "<" << getHeight() << "x" << getWidth();
  if (auto layout = getLayout())
    printer << ", " << layout;
  printer << ">";
}

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum definitions
//===----------------------------------------------------------------------===//

#include "ch3-gpu-tile-dsl/TinyTileEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd interface definitions
//===----------------------------------------------------------------------===//

#include "ch3-gpu-tile-dsl/TinyTileInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileOps.cpp.inc"
