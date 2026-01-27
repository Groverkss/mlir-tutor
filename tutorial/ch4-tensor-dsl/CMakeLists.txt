#===----------------------------------------------------------------------===#
# Chapter 4: Tensor DSL (TinyTensor dialect)
#===----------------------------------------------------------------------===#

# Generate dialect and operation definitions.
set(LLVM_TARGET_DEFINITIONS TinyTensorDialect.td)
mlir_tablegen(TinyTensorOps.h.inc -gen-op-decls)
mlir_tablegen(TinyTensorOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyTensorOpsDialect.h.inc -gen-dialect-decls -dialect=tiny_tensor)
mlir_tablegen(TinyTensorOpsDialect.cpp.inc -gen-dialect-defs -dialect=tiny_tensor)
add_public_tablegen_target(TinyTensorOpsIncGen)

# Generate pass definitions from TableGen.
set(LLVM_TARGET_DEFINITIONS TinyTensorPasses.td)
mlir_tablegen(TinyTensorPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(TinyTensorPassesIncGen)

# Create the TinyTensor dialect library.
add_mlir_library(TinyTensorDialect
  TinyTensorDialect.cpp
  LinalgToTinyTile.cpp

  DEPENDS
  TinyTensorOpsIncGen
  TinyTensorPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
  TinyDialect
  TinyTileDialect
)

# Include the current source and binary directories for generated headers.
target_include_directories(TinyTensorDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
