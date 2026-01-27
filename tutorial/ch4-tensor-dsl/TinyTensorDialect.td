//===- TinyTensorDialect.td - TinyTensor dialect definition -*- tablegen -*-===//
//
// Defines the TinyTensor dialect for the MLIR tutorial (Chapter 4).
//
// This dialect provides operations for converting between tensor types and
// pointer types, enabling interoperability between linalg and tiny_tile.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TENSOR_DIALECT
#define TINY_TENSOR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// Type constraint for tiny.ptr from ch1 (reused).
def Tiny_PtrType : Type<
  CPred<"::llvm::isa<::mlir::tiny::PtrType>($_self)">,
  "tiny.ptr type",
  "::mlir::tiny::PtrType"
>;

// Type constraint for 2D tensors of f16.
def TinyTensor_Tensor2DF16 : Type<
  And<[
    CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">,
    CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getRank() == 2">,
    CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getElementType().isF16()">
  ]>,
  "2D tensor of f16",
  "::mlir::RankedTensorType"
>;

// Type constraint for 0D tensors of f16 (scalar tensor for reduction results).
def TinyTensor_Tensor0DF16 : Type<
  And<[
    CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">,
    CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getRank() == 0">,
    CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getElementType().isF16()">
  ]>,
  "0D tensor of f16 (scalar)",
  "::mlir::RankedTensorType"
>;

//===----------------------------------------------------------------------===//
// TinyTensor dialect definition
//===----------------------------------------------------------------------===//

def TinyTensor_Dialect : Dialect {
  let name = "tiny_tensor";
  let summary = "A tensor DSL dialect bridging linalg and tiny_tile.";
  let description = [{
    The TinyTensor dialect provides operations for converting between tensor
    types and pointer types. This enables interoperability between high-level
    linalg operations and low-level tiny_tile GPU operations.

    The key operations are:
    - `to_ptr`: Convert a tensor to a pointer (for use with tiny_tile)
    - `from_ptr`: Convert a pointer to a tensor (for use with linalg)
  }];
  let cppNamespace = "::mlir::tiny_tensor";
}

//===----------------------------------------------------------------------===//
// Base operation class
//===----------------------------------------------------------------------===//

class TinyTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<TinyTensor_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def TinyTensor_ToPtrOp : TinyTensor_Op<"to_ptr", [Pure]> {
  let summary = "Convert a tensor to a pointer.";
  let description = [{
    Converts a 2D f16 tensor to a `tiny.ptr` for use with tiny_tile operations.
    This operation is the inverse of `from_ptr`.

    The tensor must be a 2D tensor of f16. The resulting pointer can be used
    with tiny_tile.load/store operations.

    Example:
    ```mlir
    %ptr = tiny_tensor.to_ptr %tensor : tensor<64x128xf16> -> !tiny.ptr
    ```
  }];

  let arguments = (ins TinyTensor_Tensor2DF16:$tensor);
  let results = (outs Tiny_PtrType:$result);

  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";

  let hasCanonicalizer = 1;
}

def TinyTensor_FromPtrOp : TinyTensor_Op<"from_ptr", [Pure]> {
  let summary = "Convert a pointer to a tensor.";
  let description = [{
    Converts a `tiny.ptr` to a 2D f16 tensor. This operation is the inverse
    of `to_ptr`.

    This operation is typically used at function boundaries where function
    arguments are pointers but internal operations use tensors.

    The dimensions of the resulting tensor must be provided as the result type
    since they cannot be inferred from the pointer type.

    Example:
    ```mlir
    %tensor = tiny_tensor.from_ptr %ptr : !tiny.ptr -> tensor<64x128xf16>
    ```
  }];

  let arguments = (ins Tiny_PtrType:$ptr);
  let results = (outs TinyTensor_Tensor2DF16:$result);

  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";

  let hasCanonicalizer = 1;
}

#endif // TINY_TENSOR_DIALECT
