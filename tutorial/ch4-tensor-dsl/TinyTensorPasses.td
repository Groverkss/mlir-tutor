//===- TinyTensorPasses.td - TinyTensor dialect passes -----*- tablegen -*-===//
//
// Defines passes for the TinyTensor dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TENSOR_PASSES
#define TINY_TENSOR_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToTinyTile : Pass<"linalg-to-tiny-tile"> {
  let summary = "Convert linalg.generic operations to tiny_tile operations.";
  let description = [{
    This pass converts linalg.generic operations with elementwise scalar
    operations to tiny_tile operations. It follows a pattern similar to
    MLIR's linalg vectorizer.

    The conversion process for elementwise operations:
    1. Identify linalg.generic ops with 2D tensor operands
    2. Convert tensor operands to pointers using tiny_tensor.to_ptr
    3. Load tiles using tiny_tile.load
    4. Convert scalar operations (arith.addf, arith.mulf, etc.) to
       tiny_tile.elementwise operations
    5. Store results using tiny_tile.store
    6. Convert pointers back to tensors using tiny_tensor.from_ptr

    Restrictions:
    - Only 2D tensors are supported for elementwise ops
    - Static tensor shapes required

    Example (elementwise):
    ```mlir
    // Before:
    %result = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>,
                       affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<64x128xf16>, tensor<64x128xf16>)
      outs(%c : tensor<64x128xf16>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %c_elem: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      linalg.yield %sum : f16
    } -> tensor<64x128xf16>

    // After:
    %a_ptr = tiny_tensor.to_ptr %a : tensor<64x128xf16> -> !tiny.ptr
    %b_ptr = tiny_tensor.to_ptr %b : tensor<64x128xf16> -> !tiny.ptr
    %c_ptr = tiny_tensor.to_ptr %c : tensor<64x128xf16> -> !tiny.ptr
    %tile_a = tiny_tile.load %a_ptr, %c0, %c0 stride %stride : ...
    %tile_b = tiny_tile.load %b_ptr, %c0, %c0 stride %stride : ...
    %tile_sum = tiny_tile.elementwise add %tile_a, %tile_b : ...
    tiny_tile.store %tile_sum, %c_ptr, %c0, %c0 stride %stride : ...
    %result = tiny_tensor.from_ptr %c_ptr : !tiny.ptr -> tensor<64x128xf16>
    ```
  }];

  let dependentDialects = [
    "mlir::tiny::TinyDialect",
    "mlir::tiny_tile::TinyTileDialect",
    "mlir::tiny_tensor::TinyTensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // TINY_TENSOR_PASSES
