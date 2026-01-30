//===- TinyTransformExtension.cpp - Transform extension for ch4 ----------===//
//
// This file registers the custom transform operations with the transform
// dialect using the extension mechanism.
//
//===----------------------------------------------------------------------===//

#include "ch4-linalg-transforms/TinyTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {

/// The TinyTransform extension registers our custom transform operations
/// with the transform dialect.
class TinyTransformExtension
    : public transform::TransformDialectExtension<TinyTransformExtension> {
public:
  // Required for MLIR's type ID system.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TinyTransformExtension)

  /// Initialize the extension by registering all transform operations.
  void init() {
    // Register our custom transform operations.
    registerTransformOps<
#define GET_OP_LIST
#include "ch4-linalg-transforms/TinyTransformOps.cpp.inc"
        >();
  }
};

} // namespace

void mlir::tiny::registerTinyTransformExtension(DialectRegistry &registry) {
  registry.addExtensions<TinyTransformExtension>();
}
