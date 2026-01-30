# Ch3: GPU Vector Tile DSL

In this chapter, we will define a "tile" based programming language, similar to
Triton and TileIR.

Tile-based programming languages are centered around the idea of a "tile": a
statically shaped multi dimensional array. For our language, our tile is
a 2-D array.

While GPUs are programmed using a SIMT programming model, Tile-based languages,
program at a level of workgroups / thread blocks. The language automatically
compiles itself down to a SIMT programming model.

In this chapter, we will implement a tile-based programming language, and
write transformations that lower it to a SIMT programming model, which can
further be compiled to PTX or to AMDGPU-ASM.

## Operations

We will define the following operations:

Tile Based Operations:

- tiny_tile.splat : Create a tile with a splatted value
- tiny_tile.elementwise : Perform an elementwise operation over two tiles
- tiny_tile.sum : Sum the entire tile down to a vector<1xf16>
- tiny_tile.load : Load a tile from a ptr
- tiny_tile.store : Store a tile to a ptr

Types:

- tiny_tile.tile : A 2-D statically shaped array with a layout

Attributes:

- tiny_tile.layout : Describes how a tile is distributed over a thread grid.

## Language

With this language, we can write gpu programs in a tile based programming model:

```python
# 1x256 tile = 256 elements
# 32 threads (1x32), each handles 8 elements
LAYOUT = Layout(thread=(1, 32), vector_size=8)
TILE_H, TILE_W = 1, 256


def dot(a: Ptr, b: Ptr, out: Ptr, M: Index, K: Index):
    """
    Given two arrays of shape MxK, we perform a dot product over them.

    c = sum(a * b)

    Each block processes a tile of 1xK elements, running over tile of 1x256
    in one go, and them computing a sum over them at the end.
    """
    # Get block ID - each block processes a different chunk of data
    bid = block_id_x()

    zero = Index.constant(0)
    tile_w = Index.constant(TILE_W)

    # Initialize accumulator tile to zeros
    acc_init = Tile.zeros(TILE_H, TILE_W, LAYOUT)

    # Run a loop over the K dimension, processing a tile of 1x256 elements at
    # once.
    @accumulate(K, tile_w, inits=[acc_init])
    def _(tile_idx: Index, acc: Tile):
        # Load a[bid,tile_idx : tile_idx + tile_w]
        a_tile = load_tile(a, bid, tile_idx, K, TILE_H, TILE_W, LAYOUT)
        b_tile = load_tile(b, bid, tile_idx, K, TILE_H, TILE_W, LAYOUT)

        # Accumulate element-wise product
        return acc + a_tile * b_tile

    # Sum final accumulated tile to vector<1xf16>
    final_acc = _[0]
    result = final_acc.sum()

    # Store this block's sum to out[block_id]
    out.store(bid, result)
```

## Layout Attribute

The layout attribute:

```mlir
#tiny_tile.layout<thread = [1, 32], vector_size = 8>
```

represents how a tile of size `<thread[0] X thread[1] * vector_size>` is
distributed over a thread grid of size: `thread[0] X thread[1]`.

It describes a mapping from thread ids to a view of the tile:

```
(thread_y, thread_x) -> tile[thread_y * num_thread_x, thread_x : thread_x + vector_size]
```

## Lowering

We implement 1 pass in this chapter:

- TinyTileToTiny:

We lower the tiny_tile operations to MLIR's tiny dialect (from ch1), using
gpu thread intrinsics to compute per-thread offsets. The key insight is that
each tile operation becomes a per-thread vector operation.

The lowering uses an interface-based pattern (`TinyTileLoweringOpInterface`).
Each tiny_tile operation implements a `convertToSIMT` method that handles its
own lowering. This approach is cleaner than having separate patterns for each
operation.

We lower:

- tiny_tile.splat -> tiny.constant (per-thread vector)
- tiny_tile.elementwise add -> tiny.addf
- tiny_tile.elementwise sub -> tiny.subf
- tiny_tile.elementwise mul -> tiny.mulf
- tiny_tile.elementwise div -> tiny.divf
- tiny_tile.load -> gpu.thread_id + tiny.load (computes per-thread offset)
- tiny_tile.store -> gpu.thread_id + tiny.store (computes per-thread offset)
- tiny_tile.sum -> tiny.sum + vector.extract + gpu.subgroup_reduce + vector.broadcast

The type converter converts `!tiny_tile.tile<HxW, layout>` to the per-thread
vector type (e.g., `vector<8xf16>` for a layout with vector_size=8).

## Playing around with the language

Switch to the `main` branch. Make sure you have the python package installed
as mentioned in ../../README.md

Open `gpu_dot_product.py` and have a look at the python dsl code. Run the file
with `python3 gpu_dot_product.py` to see the language lowering to SIMT code.

You can use this as a playground to play around with the language and see what
code is produced during the lowerings. This should get you a better
understanding of what the operations look like and what the lowerings look
like.

Try modifying the tile layout or dimensions to see how the generated code
changes.

## Exercise Implementations

Switch to `ch3-exercise` branch to access the exercises for ch3.

### Exercise 1: TinyTileToTiny Lowering Interface

Open the TinyTileDialect.td file. You will notice that some operations already
declare the `TinyTileLoweringOpInterface`:

- ElementwiseOp (already has interface + implementation)
- SplatOp (already has interface + implementation)

Your task is to:

1. Add the `TinyTileLoweringOpInterface` to the remaining operations in
   TinyTileDialect.td:
   - LoadOp
   - StoreOp
   - SumOp

2. Implement the `convertToSIMT` method for each operation in TinyTileDialect.cpp.
   Look at the existing implementations for ElementwiseOp and SplatOp for
   guidance on how to write the lowering.

   The lowerings should produce:
   - LoadOp -> gpu.thread_id + tiny.load (compute per-thread offset from layout)
   - StoreOp -> gpu.thread_id + tiny.store (compute per-thread offset from layout)
   - SumOp -> tiny.sum + vector.extract + gpu.subgroup_reduce + vector.broadcast

Once done, try testing with `test/lower-to-tiny.mlir` file, which should verify
if the lowering works as expected.

### Challenge Exercise (Optional)

Try implementing a `tiny_tile.matmul` operation that performs a tile-level
matrix multiplication. The operation should take two tiles and produce a tile
result. Think about how the layout affects the computation and how to lower
it to per-thread operations.
