"""Example: Elementwise tile operations."""

from tiny.ch3 import Ptr, Index, Layout, load_tile, store_tile, compile_and_print

# 32x32 tile = 1024 elements
# 64 threads (8x8), each handles 16 elements
LAYOUT = Layout(thread=(8, 8), vector_size=16)
TILE_H, TILE_W = 32, 32


@compile_and_print
def tile_square(a: Ptr, result: Ptr, row: Index, col: Index, stride: Index):
    """Square each element of a tile: result = a * a"""
    tile = load_tile(a, row, col, stride,
                     height=TILE_H, width=TILE_W, layout=LAYOUT)
    squared = tile * tile
    store_tile(squared, result, row, col, stride)
