"""Example: Tiled dot product with accumulate loop using tiny_tile dialect."""

from tiny.ch2 import accumulate
from tiny.ch3 import Ptr, Index, Layout, Tile, load_tile, compile_and_print

# 32x32 tile = 1024 elements
# 64 threads (8x8), each handles 16 elements
LAYOUT = Layout(thread=(8, 8), vector_size=16)
TILE_H, TILE_W = 32, 32


@compile_and_print
def dot(a: Ptr, b: Ptr, out: Ptr, n_tiles: Index, stride: Index):
    """Tiled dot product: out = sum(a[0:N] * b[0:N])

    Iterates over n_tiles tiles, loading and multiplying corresponding
    tiles from a and b, accumulating partial sums, then stores final result.
    """
    zero = Index.constant(0)
    one = Index.constant(1)
    tile_size = Index.constant(TILE_H * TILE_W)

    # Initialize accumulator tile to zeros
    acc_init = Tile.zeros(TILE_H, TILE_W, LAYOUT)

    # Accumulate loop over tiles
    @accumulate(n_tiles, one, inits=[acc_init])
    def _(tile_idx: Index, acc: Tile):
        # Compute offset for this tile
        offset = tile_idx * tile_size

        # Load tiles from a and b
        a_tile = load_tile(a, zero, offset, stride, TILE_H, TILE_W, LAYOUT)
        b_tile = load_tile(b, zero, offset, stride, TILE_H, TILE_W, LAYOUT)

        # Accumulate element-wise product
        return acc + a_tile * b_tile

    # Sum final accumulated tile to vector<1xf16>
    final_acc = _[0]
    result = final_acc.sum()

    # Store result
    out.store(zero, result)
