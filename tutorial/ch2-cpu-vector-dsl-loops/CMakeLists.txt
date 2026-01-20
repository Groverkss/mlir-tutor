#===----------------------------------------------------------------------===#
# Chapter 2: CPU Vector DSL with Loops (TinyLoop dialect)
#===----------------------------------------------------------------------===#

# Generate dialect and operation definitions from the merged TableGen file.
set(LLVM_TARGET_DEFINITIONS TinyLoopDialect.td)
mlir_tablegen(TinyLoopOps.h.inc -gen-op-decls)
mlir_tablegen(TinyLoopOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyLoopOpsDialect.h.inc -gen-dialect-decls -dialect=tiny_loop)
mlir_tablegen(TinyLoopOpsDialect.cpp.inc -gen-dialect-defs -dialect=tiny_loop)
add_public_tablegen_target(TinyLoopOpsIncGen)

# Generate pass definitions from TableGen.
set(LLVM_TARGET_DEFINITIONS TinyLoopPasses.td)
mlir_tablegen(TinyLoopPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(TinyLoopPassesIncGen)

# Create the TinyLoop dialect library.
add_mlir_library(TinyLoopDialect
  TinyLoopDialect.cpp
  TinyLoopToSCF.cpp

  DEPENDS
  TinyLoopOpsIncGen
  TinyLoopPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
)

# Include the current source and binary directories for generated headers.
target_include_directories(TinyLoopDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
