//===- TinyPasses.h - Tiny dialect passes -----------------------*- C++ -*-===//
//
// Declares passes for the Tiny dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH1_TINY_PASSES_H
#define TUTORIAL_CH1_TINY_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tiny {

// Generate pass declarations.
#define GEN_PASS_DECL
#include "ch1-cpu-vector-dsl/TinyPasses.h.inc"

// Generate pass registration.
#define GEN_PASS_REGISTRATION
#include "ch1-cpu-vector-dsl/TinyPasses.h.inc"

} // namespace tiny
} // namespace mlir

#endif // TUTORIAL_CH1_TINY_PASSES_H
