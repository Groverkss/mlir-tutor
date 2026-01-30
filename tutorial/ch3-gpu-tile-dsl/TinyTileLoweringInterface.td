//===- TinyTileLoweringInterface.td - Lowering interface ----*- tablegen -*-===//
//
// Defines the interface for lowering TinyTile ops to SIMT operations.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TILE_LOWERING_INTERFACE
#define TINY_TILE_LOWERING_INTERFACE

include "mlir/IR/OpBase.td"

def TinyTileLoweringOpInterface : OpInterface<"TinyTileLoweringOpInterface"> {
  let description = [{
    Interface for TinyTile operations that can be converted to SIMT operations.

    Operations implementing this interface provide their own conversion logic,
    which is invoked by a generic conversion pattern. The SIMT operands
    (with TileType already converted to per-thread VectorType) are passed
    to the method.
  }];

  let cppNamespace = "::mlir::tiny_tile";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Convert this operation to SIMT (per-thread) operations.

        The `simtOperands` are the operands after type conversion (TileType
        becomes per-thread VectorType). The implementation should use the
        rewriter to create replacement operations and call `rewriter.replaceOp()`.

        Returns success if conversion succeeded.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"convertToSIMT",
      /*args=*/(ins
        "::mlir::RewriterBase &":$rewriter,
        "::mlir::ValueRange":$simtOperands
      )
    >
  ];
}

#endif // TINY_TILE_LOWERING_INTERFACE
