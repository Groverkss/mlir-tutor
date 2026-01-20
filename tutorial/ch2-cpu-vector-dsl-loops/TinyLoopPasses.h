//===- TinyLoopPasses.h - TinyLoop dialect passes ---------------*- C++ -*-===//
//
// Declares passes for the TinyLoop dialect (Chapter 2).
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH2_TINY_LOOP_PASSES_H
#define TUTORIAL_CH2_TINY_LOOP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tiny_loop {

// Generate pass declarations.
#define GEN_PASS_DECL
#include "ch2-cpu-vector-dsl-loops/TinyLoopPasses.h.inc"

// Generate pass registration.
#define GEN_PASS_REGISTRATION
#include "ch2-cpu-vector-dsl-loops/TinyLoopPasses.h.inc"

} // namespace tiny_loop
} // namespace mlir

#endif // TUTORIAL_CH2_TINY_LOOP_PASSES_H
