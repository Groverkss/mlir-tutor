//===- TinyTileInterfaces.td - TinyTile interface definitions -*- tablegen -*-===//
//
// Defines the layout propagation interface for the TinyTile dialect.
//
// This interface allows operations to participate in layout propagation
// analysis, which determines how tile data is distributed across GPU threads.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TILE_INTERFACES
#define TINY_TILE_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TinyTileLayoutInterface
//===----------------------------------------------------------------------===//

def TinyTile_LayoutInterface : OpInterface<"TinyTileLayoutInterface"> {
  let description = [{
    Interface for operations that can participate in layout propagation.

    Layout propagation is a dataflow analysis that determines how tile data
    is distributed across GPU threads. Operations implement this interface
    to specify how layouts flow through them.

    The analysis works in two directions:
    - Forward: Infer result layouts from operand layouts
    - Backward: Infer operand layouts from result layouts
  }];

  let cppNamespace = "::mlir::tiny_tile";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Infer the layout for a result given the layouts of the operands.

        Returns the inferred layout, or nullptr if no layout can be inferred.
        The default implementation returns the first non-null operand layout.
      }],
      /*retTy=*/"LayoutAttr",
      /*methodName=*/"inferResultLayout",
      /*args=*/(ins "unsigned":$resultIndex,
                    "::llvm::ArrayRef<LayoutAttr>":$operandLayouts)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Infer the required layout for an operand given the layouts of results.

        Returns the required layout, or nullptr if no layout can be inferred.
        This is used for backward propagation from uses to definitions.
      }],
      /*retTy=*/"LayoutAttr",
      /*methodName=*/"inferOperandLayout",
      /*args=*/(ins "unsigned":$operandIndex,
                    "::llvm::ArrayRef<LayoutAttr>":$resultLayouts)
    >
  ];
}

#endif // TINY_TILE_INTERFACES
