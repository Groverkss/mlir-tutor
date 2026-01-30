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
