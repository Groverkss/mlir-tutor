"""Example: SPMD tiled dot product using tiny_tile dialect."""

from tiny.ch2 import accumulate
from tiny.ch3 import (
    Ptr,
    Index,
    Layout,
    Tile,
    load_tile,
    block_id_x,
    compile_and_print,
)

# 1x256 tile = 256 elements
# 32 threads (1x32), each handles 8 elements
LAYOUT = Layout(thread=(1, 32), vector_size=8)
TILE_H, TILE_W = 1, 256


@compile_and_print
def dot(a: Ptr, b: Ptr, out: Ptr, K: Index):
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
        # Load a[bid, tile_idx : tile_idx + tile_w]
        a_tile = load_tile(a, bid, tile_idx, K, TILE_H, TILE_W, LAYOUT)
        b_tile = load_tile(b, bid, tile_idx, K, TILE_H, TILE_W, LAYOUT)

        # Accumulate element-wise product
        return acc + a_tile * b_tile

    # Sum final accumulated tile to vector<1xf16>
    final_acc = _[0]
    result = final_acc.sum()

    # Store this block's sum to out[block_id]
    out.store(bid, result)
