"""Example: SPMD tiled dot product using tiny_tile dialect."""

from tiny.ch2 import accumulate
from tiny.ch3 import (
    Ptr, Index, Layout, Tile, load_tile,
    block_id_x, compile_and_print,
)

# 8x32 tile = 256 elements
# 32 threads (8x4), each handles 8 elements
LAYOUT = Layout(thread=(8, 4), vector_size=8)
TILE_H, TILE_W = 8, 32


@compile_and_print
def dot(a: Ptr, b: Ptr, out: Ptr, tiles_per_block: Index, stride: Index):
    """SPMD tiled dot product: each block computes a partial sum.

    Each GPU block processes `tiles_per_block` tiles starting from its
    block-specific offset. The partial sum is stored to out[block_id].

    For a complete dot product, launch with enough blocks to cover all data
    and reduce the partial sums on the host or with a separate kernel.
    """
    # Get block ID - each block processes a different chunk of data
    bid = block_id_x()

    zero = Index.constant(0)
    one = Index.constant(1)
    tile_size = Index.constant(TILE_H * TILE_W)

    # Compute starting offset for this block
    block_offset = bid * tiles_per_block * tile_size

    # Initialize accumulator tile to zeros
    acc_init = Tile.zeros(TILE_H, TILE_W, LAYOUT)

    # Accumulate loop over tiles assigned to this block
    @accumulate(tiles_per_block, one, inits=[acc_init])
    def _(tile_idx: Index, acc: Tile):
        # Compute offset for this tile within the block's chunk
        offset = block_offset + tile_idx * tile_size

        # Load tiles from a and b
        a_tile = load_tile(a, zero, offset, stride, TILE_H, TILE_W, LAYOUT)
        b_tile = load_tile(b, zero, offset, stride, TILE_H, TILE_W, LAYOUT)

        # Accumulate element-wise product
        return acc + a_tile * b_tile

    # Sum final accumulated tile to vector<1xf16>
    final_acc = _[0]
    result = final_acc.sum()

    # Store this block's partial sum to out[block_id]
    out.store(bid, result)
