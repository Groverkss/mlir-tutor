#===----------------------------------------------------------------------===#
# List dialect tutorial
#===----------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS ListDialect.td)
mlir_tablegen(ListOps.h.inc -gen-op-decls)
mlir_tablegen(ListOps.cpp.inc -gen-op-defs)
mlir_tablegen(ListOpsDialect.h.inc -gen-dialect-decls -dialect=list)
mlir_tablegen(ListOpsDialect.cpp.inc -gen-dialect-defs -dialect=list)
mlir_tablegen(ListTypes.h.inc -gen-typedef-decls -typedefs-dialect=list)
mlir_tablegen(ListTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=list)
add_public_tablegen_target(ListOpsIncGen)

add_mlir_library(ListDialect
  ListDialect.cpp

  DEPENDS
  ListOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

target_include_directories(ListDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
