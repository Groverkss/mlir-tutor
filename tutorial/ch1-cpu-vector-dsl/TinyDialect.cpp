//===- TinyDialect.cpp - Tiny dialect implementation ------------*- C++ -*-===//
//
// Implements the Tiny dialect.
//
//===----------------------------------------------------------------------===//

#include "ch1-cpu-vector-dsl/TinyDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tiny;

//===----------------------------------------------------------------------===//
// Tiny dialect implementation
//===----------------------------------------------------------------------===//

#include "ch1-cpu-vector-dsl/TinyOpsDialect.cpp.inc"

// Include the auto-generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "ch1-cpu-vector-dsl/TinyTypes.cpp.inc"

void TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ch1-cpu-vector-dsl/TinyOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "ch1-cpu-vector-dsl/TinyTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch1-cpu-vector-dsl/TinyOps.cpp.inc"
