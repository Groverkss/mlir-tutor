//===- TinyTilePasses.td - TinyTile dialect passes ----------*- tablegen -*-===//
//
// Defines passes for the TinyTile dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TILE_PASSES
#define TINY_TILE_PASSES

include "mlir/Pass/PassBase.td"

def TinyTileToTiny : Pass<"tiny-tile-to-tiny"> {
  let summary = "Convert tiny_tile operations to tiny dialect operations.";
  let description = [{
    This pass converts tiny_tile operations to per-thread tiny vector operations.
    All tiles must have layout attributes.

    The conversion uses `gpu.thread_id` to get the thread position and computes
    per-thread offsets based on the layout. Each tile operation becomes a vector
    operation on the thread's local data.

    The lowering includes:
    - `tiny_tile.elementwise` -> `tiny.addf/subf/mulf/divf` on per-thread vectors
    - `tiny_tile.load` -> `tiny.load` with computed per-thread offset
    - `tiny_tile.store` -> `tiny.store` with computed per-thread offset
    - `tiny_tile.sum` -> `tiny.sum` + `gpu.all_reduce` + `vector.broadcast` for cross-thread reduction
  }];

  let dependentDialects = [
    "mlir::tiny::TinyDialect",
    "mlir::gpu::GPUDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

#endif // TINY_TILE_PASSES
