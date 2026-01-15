//===- tutorial-opt.cpp - Tutorial MLIR Optimizer Driver ------------------===//
//
// A simple mlir-opt like tool for the tutorial.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "list/ListDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all MLIR dialects.
  mlir::registerAllDialects(registry);

  // Register all MLIR passes.
  mlir::registerAllPasses();

  // Register all extensions.
  mlir::registerAllExtensions(registry);

  // Register tutorial dialects.
  registry.insert<mlir::tutorial::list::ListDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial optimizer driver\n", registry));
}
