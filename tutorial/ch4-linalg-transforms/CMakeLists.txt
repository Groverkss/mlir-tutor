#===----------------------------------------------------------------------===#
# Chapter 4: Linalg and Transform Dialect
#===----------------------------------------------------------------------===#

# Generate transform operation definitions from TableGen.
set(LLVM_TARGET_DEFINITIONS TinyTransformOps.td)
mlir_tablegen(TinyTransformOps.h.inc -gen-op-decls)
mlir_tablegen(TinyTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TinyTransformOpsIncGen)

# Create the TinyTransform extension library.
add_mlir_library(TinyTransformExtension
  TinyTransformOps.cpp
  TinyTransformExtension.cpp

  DEPENDS
  TinyTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTilingInterface
  MLIRPass
)

# Include the current source and binary directories for generated headers.
target_include_directories(TinyTransformExtension PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
