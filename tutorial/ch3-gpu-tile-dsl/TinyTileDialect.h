//===- TinyTileDialect.h - TinyTile dialect header --------------*- C++ -*-===//
//
// Declares the TinyTile dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH3_TINY_TILE_DIALECT_H
#define TUTORIAL_CH3_TINY_TILE_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the Tiny dialect for !tiny.ptr type.
#include "ch1-cpu-vector-dsl/TinyDialect.h"

// Include the auto-generated dialect declaration.
#include "ch3-gpu-tile-dsl/TinyTileOpsDialect.h.inc"

// Include the auto-generated enum declarations (must come before attrs).
#include "ch3-gpu-tile-dsl/TinyTileEnums.h.inc"

// Include the auto-generated attribute declarations.
#define GET_ATTRDEF_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileAttrs.h.inc"

// Include the auto-generated type declarations.
#define GET_TYPEDEF_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileTypes.h.inc"

// Include the auto-generated interface declarations.
#include "ch3-gpu-tile-dsl/TinyTileInterfaces.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "ch3-gpu-tile-dsl/TinyTileOps.h.inc"

#endif // TUTORIAL_CH3_TINY_TILE_DIALECT_H
