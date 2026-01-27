//===- TinyPasses.td - Tiny dialect passes -----------------*- tablegen -*-===//
//
// Defines passes for the Tiny dialect.
// 
// Reference: https://mlir.llvm.org/docs/PassManagement/#tablegen-specification
//
//===----------------------------------------------------------------------===//

#ifndef TINY_PASSES
#define TINY_PASSES

include "mlir/Pass/PassBase.td"

def TinyToArith : Pass<"tiny-to-arith"> {
  let summary = "Lower Tiny arithmetic operations to arith dialect.";
  let description = [{
    This pass lowers Tiny dialect arithmetic operations to equivalent operations
    in the arith dialect. Memory operations (load/store) and the ptr type are
    NOT converted by this pass - use --tiny-to-llvm for that.

    The lowering includes:
    - `tiny.constant` -> `arith.constant`
    - `tiny.addf/subf/mulf/divf` -> `arith.addf/subf/mulf/divf`
    - `tiny.addi/subi/muli/divi` -> `arith.addi/subi/muli/divsi`

    Example:
    ```mlir
    %0 = tiny.addf %a, %b : vector<4xf16>
    // Becomes:
    %0 = arith.addf %a, %b : vector<4xf16>
    ```
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

def TinyToLLVM : Pass<"tiny-to-llvm"> {
  let summary = "Lower Tiny memory operations and ptr type to LLVM dialect.";
  let description = [{
    This pass lowers Tiny dialect memory operations and the ptr type to
    equivalent operations in the LLVM dialect.

    The lowering includes:
    - `!tiny.ptr` type -> `!llvm.ptr`
    - `tiny.load %ptr, %offset` -> GEP to compute byte address, then `llvm.load`
    - `tiny.store %val, %ptr, %offset` -> GEP to compute byte address, then `llvm.store`

    The offset in tiny.load/store is in f16 elements. The lowering converts this
    to a byte offset by using GEP with f16 element type:
    ```mlir
    %0 = tiny.load %ptr, %offset : vector<4xf16>
    // Becomes:
    %gep = llvm.getelementptr %ptr[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %0 = llvm.load %gep : !llvm.ptr -> vector<4xf16>
    ```

    Note: Run --tiny-to-arith first to convert arithmetic operations.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // TINY_PASSES
