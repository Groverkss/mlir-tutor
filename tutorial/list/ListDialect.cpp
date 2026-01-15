//===- ListDialect.cpp - List dialect implementation ------------*- C++ -*-===//
//
// Implements the list dialect.
//
//===----------------------------------------------------------------------===//

#include "list/ListDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tutorial::list;

#include "list/ListOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "list/ListTypes.cpp.inc"

void ListDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "list/ListTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "list/ListOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "list/ListOps.cpp.inc"
