//===- TinyTilePasses.td - TinyTile dialect passes ----------*- tablegen -*-===//
//
// Defines passes for the TinyTile dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TILE_PASSES
#define TINY_TILE_PASSES

include "mlir/Pass/PassBase.td"

def TinyTileLayoutPropagation : Pass<"tiny-tile-layout-propagation"> {
  let summary = "Propagate layout attributes through tiny_tile operations.";
  let description = [{
    This pass propagates layout attributes from tiles with layouts to tiles
    without layouts. It uses the TinyTileLayoutInterface to determine how
    layouts flow through operations.

    The propagation works in two directions:
    - Forward: Infer result layouts from operand layouts
    - Backward: Infer operand layouts from result layouts

    After this pass, all tile types should have layouts. The pass will emit
    an error if any tile remains without a layout.

    Example:
    ```mlir
    // Before:
    %a = tiny_tile.load %ptr, %r, %c stride %s : !tiny_tile.tile<64x128, #layout>
    %b = tiny_tile.elementwise add %a, %a : !tiny_tile.tile<64x128>  // no layout

    // After:
    %a = tiny_tile.load %ptr, %r, %c stride %s : !tiny_tile.tile<64x128, #layout>
    %b = tiny_tile.elementwise add %a, %a : !tiny_tile.tile<64x128, #layout>
    ```
  }];
}

def TinyTileToTiny : Pass<"tiny-tile-to-tiny"> {
  let summary = "Convert tiny_tile operations to tiny dialect operations.";
  let description = [{
    This pass converts tiny_tile operations to per-thread tiny vector operations.
    It requires all tiles to have layout attributes (run layout propagation first).

    The conversion uses `gpu.thread_id` to get the thread position and computes
    per-thread offsets based on the layout. Each tile operation becomes a vector
    operation on the thread's local data.

    The lowering includes:
    - `tiny_tile.elementwise` -> `tiny.addf/subf/mulf/divf` on per-thread vectors
    - `tiny_tile.load` -> `tiny.load` with computed per-thread offset
    - `tiny_tile.store` -> `tiny.store` with computed per-thread offset
    - `tiny_tile.reduce` -> `tiny.sum` + `gpu.all_reduce` for cross-thread reduction

    This pass should be run inside a `gpu.launch` region.
  }];

  let dependentDialects = [
    "mlir::tiny::TinyDialect",
    "mlir::gpu::GPUDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

#endif // TINY_TILE_PASSES
