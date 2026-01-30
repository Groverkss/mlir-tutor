//===- TinyTileDialect.td - TinyTile dialect definition ----*- tablegen -*-===//
//
// Defines the TinyTile dialect for the MLIR tutorial (Chapter 3).
//
// This dialect provides tile-based operations for GPU programming.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TILE_DIALECT
#define TINY_TILE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Type constraint for tiny.ptr from ch1 (for load/store operations).
// This references the C++ type defined in ch1-cpu-vector-dsl/TinyDialect.h.
def Tiny_PtrType : Type<
  CPred<"::llvm::isa<::mlir::tiny::PtrType>($_self)">,
  "tiny.ptr type",
  "::mlir::tiny::PtrType"
>;

//===----------------------------------------------------------------------===//
// TinyTile dialect definition
//===----------------------------------------------------------------------===//

def TinyTile_Dialect : Dialect {
  let name = "tiny_tile";
  let summary = "A tile-based GPU DSL dialect";
  let description = [{
    The TinyTile dialect provides tile-based operations for GPU programming.
    Tiles are 2D arrays with static dimensions that are distributed across
    GPU threads using a layout attribute.
  }];
  let cppNamespace = "::mlir::tiny_tile";

  // Enable default type/attribute printer/parser for custom types.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Layout Attribute
//===----------------------------------------------------------------------===//

def TinyTile_LayoutAttr : AttrDef<TinyTile_Dialect, "Layout"> {
  let mnemonic = "layout";
  let summary = "Describes how tile data is distributed across GPU threads.";
  let description = [{
    The layout attribute specifies how a tile's data is distributed across
    GPU threads.

    Parameters:
    - `thread`: [rows, cols] - the 2D grid of threads
    - `vectorSize`: number of elements each thread owns

    The total elements in a tile must equal thread[0] * thread[1] * vectorSize.

    Example:
    ```mlir
    #tiny_tile.layout<thread = [16, 4], vector_size = 8>
    ```
    This means 64 threads (16x4 grid), each owning 8 elements (512 total).
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$thread,
    "int64_t":$vectorSize
  );

  let assemblyFormat = "`<` `thread` `=` `[` $thread `]` `,` `vector_size` `=` $vectorSize `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the total number of threads.
    int64_t getNumThreads() const {
      int64_t result = 1;
      for (int64_t dim : getThread())
        result *= dim;
      return result;
    }

    /// Returns the total number of elements covered by this layout.
    int64_t getNumElements() const {
      return getNumThreads() * getVectorSize();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Tile Type
//===----------------------------------------------------------------------===//

def TinyTile_TileType : TypeDef<TinyTile_Dialect, "Tile"> {
  let mnemonic = "tile";
  let summary = "A 2D tile of f16 data with static dimensions and layout.";
  let description = [{
    The tile type represents a 2D block of f16 data with static dimensions
    and a layout describing how it is distributed across GPU threads.
    Dimensions must be powers of 2.

    Example:
    ```mlir
    !tiny_tile.tile<64x128, #tiny_tile.layout<thread=[16,4], vector_size=8>>
    ```
  }];

  let parameters = (ins
    "int64_t":$height,
    "int64_t":$width,
    "LayoutAttr":$layout
  );

  // Use custom parser/printer because the lexer treats "64x128" as a dimension
  // list token. We want `<64x128, layout>` syntax.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the total number of elements in the tile.
    int64_t getNumElements() const { return getHeight() * getWidth(); }

    /// Returns the per-thread vector type.
    VectorType getPerThreadVectorType() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type constraint for tile types
//===----------------------------------------------------------------------===//

def TinyTile_Tile : Type<CPred<"::llvm::isa<::mlir::tiny_tile::TileType>($_self)">,
                         "tile type",
                         "::mlir::tiny_tile::TileType">;

//===----------------------------------------------------------------------===//
// Base operation class
//===----------------------------------------------------------------------===//

class TinyTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<TinyTile_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Elementwise operation kind enum
//===----------------------------------------------------------------------===//

def TinyTile_EWKind_Add : I32EnumAttrCase<"add", 0>;
def TinyTile_EWKind_Sub : I32EnumAttrCase<"sub", 1>;
def TinyTile_EWKind_Mul : I32EnumAttrCase<"mul", 2>;
def TinyTile_EWKind_Div : I32EnumAttrCase<"div", 3>;

def TinyTile_ElementwiseKind : I32EnumAttr<
    "ElementwiseKind",
    "Elementwise operation kind",
    [TinyTile_EWKind_Add, TinyTile_EWKind_Sub,
     TinyTile_EWKind_Mul, TinyTile_EWKind_Div]> {
  let cppNamespace = "::mlir::tiny_tile";
  let genSpecializedAttr = 0;
}

def TinyTile_ElementwiseKindAttr :
    EnumAttr<TinyTile_Dialect, TinyTile_ElementwiseKind, "ew_kind">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def TinyTile_ElementwiseOp : TinyTile_Op<"elementwise", [Pure]> {
  let summary = "Element-wise binary operation on tiles.";
  let description = [{
    Performs an element-wise binary operation on two tiles.
    The kind attribute specifies the operation (add, sub, mul, div).

    All operands and the result must have the same tile dimensions and layout.

    Example:
    ```mlir
    %c = tiny_tile.elementwise add %a, %b
        : !tiny_tile.tile<64x128, #tiny_tile.layout<...>>,
          !tiny_tile.tile<64x128, #tiny_tile.layout<...>>
       -> !tiny_tile.tile<64x128, #tiny_tile.layout<...>>
    ```
  }];

  let arguments = (ins
    TinyTile_ElementwiseKindAttr:$kind,
    TinyTile_Tile:$lhs,
    TinyTile_Tile:$rhs
  );
  let results = (outs TinyTile_Tile:$result);

  let assemblyFormat = "$kind $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
}

def TinyTile_ReduceOp : TinyTile_Op<"reduce", [Pure]> {
  let summary = "Reduce all elements of a tile to a scalar.";
  let description = [{
    Reduces all elements of a tile using addition, producing a scalar f16.
    This performs a full reduction across all dimensions and all threads.

    Example:
    ```mlir
    %sum = tiny_tile.reduce %tile
        : !tiny_tile.tile<64x128, #tiny_tile.layout<...>> -> f16
    ```
  }];

  let arguments = (ins TinyTile_Tile:$input);
  let results = (outs F16:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def TinyTile_LoadOp : TinyTile_Op<"load"> {
  let summary = "Load a tile from memory.";
  let description = [{
    Loads a tile from memory at the given pointer with row and column offsets.
    The stride specifies the number of f16 elements between rows in memory.
    The result type must include a layout.

    Example:
    ```mlir
    %tile = tiny_tile.load %ptr, %row, %col stride %s
        : !tiny.ptr -> !tiny_tile.tile<64x128, #tiny_tile.layout<...>>
    ```
  }];

  let arguments = (ins
    Tiny_PtrType:$ptr,
    Index:$row,
    Index:$col,
    Index:$stride
  );
  let results = (outs TinyTile_Tile:$result);

  let assemblyFormat = "$ptr `,` $row `,` $col `stride` $stride attr-dict `:` type($ptr) `->` type($result)";
}

def TinyTile_StoreOp : TinyTile_Op<"store"> {
  let summary = "Store a tile to memory.";
  let description = [{
    Stores a tile to memory at the given pointer with row and column offsets.
    The stride specifies the number of f16 elements between rows in memory.

    Example:
    ```mlir
    tiny_tile.store %tile, %ptr, %row, %col stride %s
        : !tiny_tile.tile<64x128, #tiny_tile.layout<...>>, !tiny.ptr
    ```
  }];

  let arguments = (ins
    TinyTile_Tile:$value,
    Tiny_PtrType:$ptr,
    Index:$row,
    Index:$col,
    Index:$stride
  );

  let assemblyFormat = "$value `,` $ptr `,` $row `,` $col `stride` $stride attr-dict `:` type($value) `,` type($ptr)";
}

def TinyTile_SplatOp : TinyTile_Op<"splat", [Pure]> {
  let summary = "Create a tile with all elements set to the same value.";
  let description = [{
    Creates a tile where all elements have the same f16 value.
    The result type must include a layout.

    Example:
    ```mlir
    %tile = tiny_tile.splat 1.0
        : !tiny_tile.tile<32x16, #tiny_tile.layout<...>>
    ```
  }];

  let arguments = (ins F64Attr:$splatValue);
  let results = (outs TinyTile_Tile:$result);

  let assemblyFormat = "$splatValue attr-dict `:` type($result)";
}

#endif // TINY_TILE_DIALECT
