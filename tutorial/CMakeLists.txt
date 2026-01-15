set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(tutorial-opt
  tutorial-opt.cpp
)

llvm_update_compile_flags(tutorial-opt)

target_link_libraries(tutorial-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
