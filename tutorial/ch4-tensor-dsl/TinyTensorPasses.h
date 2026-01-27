//===- TinyTensorPasses.h - TinyTensor dialect passes -----------*- C++ -*-===//
//
// Declares passes for the TinyTensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH4_TINY_TENSOR_PASSES_H
#define TUTORIAL_CH4_TINY_TENSOR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tiny_tensor {

// Generate pass declarations.
#define GEN_PASS_DECL
#include "ch4-tensor-dsl/TinyTensorPasses.h.inc"

// Generate pass registration.
#define GEN_PASS_REGISTRATION
#include "ch4-tensor-dsl/TinyTensorPasses.h.inc"

} // namespace tiny_tensor
} // namespace mlir

#endif // TUTORIAL_CH4_TINY_TENSOR_PASSES_H
