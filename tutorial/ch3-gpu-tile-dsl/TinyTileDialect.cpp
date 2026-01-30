//===- TinyTileDialect.cpp - TinyTile dialect implementation -----*- C++ -*-===//
//
// Implements the TinyTile dialect, including verifiers.
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

  // Layout is required.
  if (!layout) {
    return emitError() << "tile type requires a layout";
  }

  // Verify tile elements match layout elements.
  int64_t tileElements = height * width;
  int64_t layoutElements = layout.getNumElements();
  if (tileElements != layoutElements) {
    return emitError() << "tile has " << tileElements << " elements but layout covers "
                       << layoutElements << " elements";
  }

  return success();
}

VectorType TileType::getPerThreadVectorType() const {
  auto layout = getLayout();
  auto elementType = Float16Type::get(getContext());
  return VectorType::get({layout.getVectorSize()}, elementType);
}

//===----------------------------------------------------------------------===//
// ElementwiseOp verification
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

  // All operands and result must have the same layout.
  if (lhsType.getLayout() != rhsType.getLayout()) {
    return emitOpError() << "operands must have the same layout";
  }
  if (lhsType.getLayout() != resultType.getLayout()) {
    return emitOpError() << "result must have the same layout as operands";
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

/// Parse a tile type: `<` HxW `,` layout `>`
/// Example: <64x128, #tiny_tile.layout<thread=[16,4], vector_size=8>>
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

  // Parse required comma and layout attribute.
  if (parser.parseComma())
    return Type();

  LayoutAttr layout;
  if (parser.parseAttribute(layout))
    return Type();

  if (parser.parseGreater())
    return Type();

  return TileType::get(parser.getContext(), dims[0], dims[1], layout);
}

/// Print a tile type: `<` HxW `,` layout `>`
void TileType::print(AsmPrinter &printer) const {
  printer << "<" << getHeight() << "x" << getWidth() << ", " << getLayout() << ">";
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
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileOps.cpp.inc"
