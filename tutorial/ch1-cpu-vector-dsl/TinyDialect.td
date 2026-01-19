//===- TinyDialect.td - Tiny dialect definition ------------*- tablegen -*-===//
//
// Defines the Tiny dialect for the MLIR tutorial.
// This dialect provides simple vector operations on f16 vectors and index 
// computations.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_DIALECT
#define TINY_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tiny dialect definition
//===----------------------------------------------------------------------===//

def Tiny_Dialect : Dialect {
  let name = "tiny";
  let summary = "A simple vector DSL dialect for the MLIR tutorial.";
  let description = [{
    The Tiny dialect provides simple vector operations on f16 vectors and
    index computations.
  }];
  let cppNamespace = "::mlir::tiny";

  // Enable default type printer/parser for custom types.
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

// Base class for Tiny dialect types.
class Tiny_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tiny_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Pointer type for memory operations.
// This is an opaque pointer that will be lowered to memref<?xf16>.
def Tiny_PtrType : Tiny_Type<"Ptr", "ptr"> {
  let summary = "Pointer to f16 data in memory.";
  let description = [{
    The `tiny.ptr` type represents a pointer to f16 data in memory.

    Example:
    ```mlir
    %ptr : !tiny.ptr
    ```
  }];
}


//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class Tiny_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tiny_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// Constraint for vector<Nxf16> where N is a power of 2.
// Uses VectorOfRankAndType from CommonTypeConstraints.td (included via OpBase.td)
// with an additional power-of-2 size check.
def Tiny_VectorF16 : Type<
  And<[
    VectorOfRankAndType<[1], [F16]>.predicate,
    CPred<"::llvm::isPowerOf2_64("
          "::llvm::cast<::mlir::VectorType>($_self).getDimSize(0))">
  ]>,
  "vector of f16 with power-of-2 size",
  "::mlir::VectorType"
>;


//===----------------------------------------------------------------------===//
// Constant operation
//===----------------------------------------------------------------------===//

def Tiny_ConstantOp : Tiny_Op<"constant", [Pure,
                                           AllTypesMatch<["value", "result"]>]> {
  let summary = "Creates a constant vector or index value.";
  let description = [{
    The `tiny.constant` operation creates a constant value which can be either
    a vector<Nxf16> or an index.

    Examples:
    ```mlir
    %0 = tiny.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf16>
    %1 = tiny.constant 42 : index
    ```
  }];

  // TypedAttrInterface allows the type to be inferred from the attribute.
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyTypeOf<[Tiny_VectorF16, Index]>:$result);

  // The attribute itself contains the type, so no need to print it separately.
  let assemblyFormat = "attr-dict $value";
}

//===----------------------------------------------------------------------===//
// Vector binary operations base class
//===----------------------------------------------------------------------===//

class Tiny_VectorBinaryOp<string mnemonic, list<Trait> traits = []> :
    Tiny_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType], traits)> {

  let arguments = (ins Tiny_VectorF16:$lhs, Tiny_VectorF16:$rhs);
  let results = (outs Tiny_VectorF16:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Vector arithmetic operations
//===----------------------------------------------------------------------===//

def Tiny_AddFOp : Tiny_VectorBinaryOp<"addf", [Commutative]> {
  let summary = "Element-wise vector addition.";
  let description = [{
    Performs element-wise addition of two f16 vectors.

    Example:
    ```mlir
    %0 = tiny.addf %a, %b : vector<4xf16>
    ```
  }];
}

def Tiny_SubFOp : Tiny_VectorBinaryOp<"subf"> {
  let summary = "Element-wise vector subtraction.";
  let description = [{
    Performs element-wise subtraction of two f16 vectors.

    Example:
    ```mlir
    %0 = tiny.subf %a, %b : vector<4xf16>
    ```
  }];
}

def Tiny_MulFOp : Tiny_VectorBinaryOp<"mulf", [Commutative]> {
  let summary = "Element-wise vector multiplication.";
  let description = [{
    Performs element-wise multiplication of two f16 vectors.

    Example:
    ```mlir
    %0 = tiny.mulf %a, %b : vector<4xf16>
    ```
  }];
}

def Tiny_DivFOp : Tiny_VectorBinaryOp<"divf"> {
  let summary = "Element-wise vector division.";
  let description = [{
    Performs element-wise division of two f16 vectors.

    Example:
    ```mlir
    %0 = tiny.divf %a, %b : vector<4xf16>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Vector reduction operations
//===----------------------------------------------------------------------===//

// Constraint for vector<1xf16> (result type for sum operation).
// Reuses Tiny_VectorF16 predicate and adds a size=1 constraint.
def Tiny_Vector1F16 : Type<
  And<[
    Tiny_VectorF16.predicate,
    CPred<"::llvm::cast<::mlir::VectorType>($_self).getDimSize(0) == 1">
  ]>,
  "vector<1xf16>",
  "::mlir::VectorType"
>;

def Tiny_SumOp : Tiny_Op<"sum", [Pure]> {
  let summary = "Compute the sum of all elements in a vector.";
  let description = [{
    Computes the sum of all elements in the input f16 vector and returns
    the result as a vector<1xf16>.

    Example:
    ```mlir
    %result = tiny.sum %input : vector<4xf16> -> vector<1xf16>
    ```
  }];

  let arguments = (ins Tiny_VectorF16:$input);
  let results = (outs Tiny_Vector1F16:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Index binary operations base class
//===----------------------------------------------------------------------===//

class Tiny_IndexBinaryOp<string mnemonic, list<Trait> traits = []> :
    Tiny_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType], traits)> {

  let arguments = (ins Index:$lhs, Index:$rhs);
  let results = (outs Index:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// Index arithmetic operations
//===----------------------------------------------------------------------===//

def Tiny_AddIOp : Tiny_IndexBinaryOp<"addi", [Commutative]> {
  let summary = "Index addition.";
  let description = [{
    Performs addition of two index values.

    Example:
    ```mlir
    %0 = tiny.addi %a, %b
    ```
  }];
}

def Tiny_SubIOp : Tiny_IndexBinaryOp<"subi"> {
  let summary = "Index subtraction.";
  let description = [{
    Performs subtraction of two index values.

    Example:
    ```mlir
    %0 = tiny.subi %a, %b
    ```
  }];
}

def Tiny_MulIOp : Tiny_IndexBinaryOp<"muli", [Commutative]> {
  let summary = "Index multiplication.";
  let description = [{
    Performs multiplication of two index values.

    Example:
    ```mlir
    %0 = tiny.muli %a, %b
    ```
  }];
}

def Tiny_DivIOp : Tiny_IndexBinaryOp<"divi"> {
  let summary = "Index division.";
  let description = [{
    Performs signed division of two index values.

    Example:
    ```mlir
    %0 = tiny.divi %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

def Tiny_LoadOp : Tiny_Op<"load", [Pure]> {
  let summary = "Load a vector from memory.";
  let description = [{
    Loads a vector<Nxf16> from memory at the given pointer and element offset.
    The offset is in f16 elements (not bytes).

    Example:
    ```mlir
    %0 = tiny.load %ptr, %offset : vector<4xf16>
    ```
  }];

  let arguments = (ins Tiny_PtrType:$ptr, Index:$offset);
  let results = (outs Tiny_VectorF16:$result);

  let assemblyFormat = "$ptr `,` $offset attr-dict `:` type($result)";
}

def Tiny_StoreOp : Tiny_Op<"store"> {
  let summary = "Store a vector to memory.";
  let description = [{
    Stores a vector<Nxf16> to memory at the given pointer and element offset.
    The offset is in f16 elements (not bytes).

    Example:
    ```mlir
    tiny.store %vec, %ptr, %offset : vector<4xf16>
    ```
  }];

  let arguments = (ins Tiny_VectorF16:$value, Tiny_PtrType:$ptr, Index:$offset);
  let results = (outs);

  let assemblyFormat = "$value `,` $ptr `,` $offset attr-dict `:` type($value)";
}


#endif // TINY_DIALECT
