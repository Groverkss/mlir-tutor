#===----------------------------------------------------------------------===#
# Chapter 1: CPU Vector DSL (Tiny dialect)
#===----------------------------------------------------------------------===#

# Generate dialect, type, and operation definitions from TableGen.
set(LLVM_TARGET_DEFINITIONS TinyDialect.td)
mlir_tablegen(TinyOps.h.inc -gen-op-decls)
mlir_tablegen(TinyOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyOpsDialect.h.inc -gen-dialect-decls -dialect=tiny)
mlir_tablegen(TinyOpsDialect.cpp.inc -gen-dialect-defs -dialect=tiny)
mlir_tablegen(TinyTypes.h.inc -gen-typedef-decls -typedefs-dialect=tiny)
mlir_tablegen(TinyTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tiny)
add_public_tablegen_target(TinyOpsIncGen)

# Generate pass definitions from TableGen.
set(LLVM_TARGET_DEFINITIONS TinyPasses.td)
mlir_tablegen(TinyPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(TinyPassesIncGen)

# Create the Tiny dialect library.
add_mlir_library(TinyDialect
  TinyDialect.cpp
  TinyToArith.cpp
  TinyToLLVM.cpp

  DEPENDS
  TinyOpsIncGen
  TinyPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

# Include the current source and binary directories for generated headers.
target_include_directories(TinyDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
