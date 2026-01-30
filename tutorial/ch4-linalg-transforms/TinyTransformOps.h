//===- TinyTransformOps.h - Transform ops for ch4 tutorial ------*- C++ -*-===//
//
// This file declares custom transform dialect operations for the tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH4_TINY_TRANSFORM_OPS_H
#define TUTORIAL_CH4_TINY_TRANSFORM_OPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace tiny {

/// Register the TinyTransform extension with the transform dialect.
/// Call this in your tool's main() before running any passes.
void registerTinyTransformExtension(DialectRegistry &registry);

} // namespace tiny
} // namespace mlir

//===----------------------------------------------------------------------===//
// Generated Op Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch4-linalg-transforms/TinyTransformOps.h.inc"

#endif // TUTORIAL_CH4_TINY_TRANSFORM_OPS_H
