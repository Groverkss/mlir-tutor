//===- ListDialect.h - List dialect declarations ----------------*- C++ -*-===//
//
// Declares the list dialect, type, and operations.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_LIST_LIST_DIALECT_H
#define TUTORIAL_LIST_LIST_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "list/ListOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "list/ListTypes.h.inc"

#define GET_OP_CLASSES
#include "list/ListOps.h.inc"

#endif // TUTORIAL_LIST_LIST_DIALECT_H
