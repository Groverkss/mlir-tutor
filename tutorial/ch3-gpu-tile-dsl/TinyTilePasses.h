//===- TinyTilePasses.h - TinyTile dialect passes ---------------*- C++ -*-===//
//
// Declares passes for the TinyTile dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH3_TINY_TILE_PASSES_H
#define TUTORIAL_CH3_TINY_TILE_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tiny_tile {

// Generate pass declarations.
#define GEN_PASS_DECL
#include "ch3-gpu-tile-dsl/TinyTilePasses.h.inc"

// Generate pass registration.
#define GEN_PASS_REGISTRATION
#include "ch3-gpu-tile-dsl/TinyTilePasses.h.inc"

} // namespace tiny_tile
} // namespace mlir

#endif // TUTORIAL_CH3_TINY_TILE_PASSES_H
