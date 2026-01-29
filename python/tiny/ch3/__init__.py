"""Chapter 3: TinyTile dialect DSL (GPU tiles)."""

from ..ch1 import Index, Ptr
from .dsl import Layout, Tile, load_tile, store_tile, print_ir, compile_and_print

__all__ = [
    "Index", "Ptr",
    "Layout", "Tile",
    "load_tile", "store_tile",
    "print_ir", "compile_and_print",
]
