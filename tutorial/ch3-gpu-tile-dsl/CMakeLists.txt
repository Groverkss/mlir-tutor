#===----------------------------------------------------------------------===#
# Chapter 3: GPU Tile DSL (TinyTile dialect)
#===----------------------------------------------------------------------===#

# Generate dialect, types, attributes, and operation definitions.
set(LLVM_TARGET_DEFINITIONS TinyTileDialect.td)
mlir_tablegen(TinyTileOps.h.inc -gen-op-decls)
mlir_tablegen(TinyTileOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyTileOpsDialect.h.inc -gen-dialect-decls -dialect=tiny_tile)
mlir_tablegen(TinyTileOpsDialect.cpp.inc -gen-dialect-defs -dialect=tiny_tile)
mlir_tablegen(TinyTileTypes.h.inc -gen-typedef-decls -typedefs-dialect=tiny_tile)
mlir_tablegen(TinyTileTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tiny_tile)
mlir_tablegen(TinyTileAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=tiny_tile)
mlir_tablegen(TinyTileAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tiny_tile)
mlir_tablegen(TinyTileEnums.h.inc -gen-enum-decls)
mlir_tablegen(TinyTileEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TinyTileOpsIncGen)

# Generate pass definitions from TableGen.
set(LLVM_TARGET_DEFINITIONS TinyTilePasses.td)
mlir_tablegen(TinyTilePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(TinyTilePassesIncGen)

# Create the TinyTile dialect library.
add_mlir_library(TinyTileDialect
  TinyTileDialect.cpp
  TinyTileToTiny.cpp

  DEPENDS
  TinyTileOpsIncGen
  TinyTilePassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUDialect
  MLIRArithDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  TinyDialect
)

# Include the current source and binary directories for generated headers.
target_include_directories(TinyTileDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)
