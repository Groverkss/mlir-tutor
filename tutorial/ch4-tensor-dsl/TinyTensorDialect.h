//===- TinyTensorDialect.h - TinyTensor dialect header ----------*- C++ -*-===//
//
// Declares the TinyTensor dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH4_TINY_TENSOR_DIALECT_H
#define TUTORIAL_CH4_TINY_TENSOR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the Tiny dialect for !tiny.ptr type.
#include "ch1-cpu-vector-dsl/TinyDialect.h"

// Include the auto-generated dialect declaration.
#include "ch4-tensor-dsl/TinyTensorOpsDialect.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "ch4-tensor-dsl/TinyTensorOps.h.inc"

#endif // TUTORIAL_CH4_TINY_TENSOR_DIALECT_H
