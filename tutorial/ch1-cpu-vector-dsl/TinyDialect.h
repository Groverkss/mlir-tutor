//===- TinyDialect.h - Tiny dialect header ----------------------*- C++ -*-===//
//
// Declares the Tiny dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH1_TINY_DIALECT_H
#define TUTORIAL_CH1_TINY_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated dialect declaration.
#include "ch1-cpu-vector-dsl/TinyOpsDialect.h.inc"

// Include the auto-generated type declarations.
#define GET_TYPEDEF_CLASSES
#include "ch1-cpu-vector-dsl/TinyTypes.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "ch1-cpu-vector-dsl/TinyOps.h.inc"

#endif // TUTORIAL_CH1_TINY_DIALECT_H
