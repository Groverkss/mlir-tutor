"""Chapter 3: TinyTile dialect DSL (GPU tiles)."""

from ..ch1 import Index, Ptr
from .dsl import (
    Layout,
    Tile,
    load_tile,
    store_tile,
    block_id_x,
    block_id_y,
    block_id_z,
    print_ir,
    compile_and_print,
)
