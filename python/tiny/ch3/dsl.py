"""DSL for TinyTile dialect (Chapter 3)."""

from mlir.ir import Value, Type, Operation, Attribute, IndexType, F16Type, F64Type, FloatAttr
from ..ch1.dsl import Index, Ptr
from ..compiler import MLIRModule, TutorialOpt


class Layout:
    """Thread distribution layout for tiles."""

    def __init__(self, thread: tuple[int, int], vector_size: int):
        self.thread = thread
        self.vector_size = vector_size

    def num_elements(self) -> int:
        return self.thread[0] * self.thread[1] * self.vector_size


class Tile:
    """Wrapper for !tiny_tile.tile SSA values."""
    _value: Value
    _height: int
    _width: int
    _layout: Layout | None

    @staticmethod
    def _wrap(value: Value, height: int, width: int, layout: Layout = None) -> "Tile":
        t = Tile()
        t._value = value
        t._height = height
        t._width = width
        t._layout = layout
        return t

    @staticmethod
    def _get_type(height: int, width: int, layout: Layout = None) -> Type:
        if layout:
            return Type.parse(
                f"!tiny_tile.tile<{height}x{width}, "
                f"#tiny_tile.layout<thread = [{layout.thread[0]}, {layout.thread[1]}], "
                f"vector_size = {layout.vector_size}>>"
            )
        return Type.parse(f"!tiny_tile.tile<{height}x{width}>")

    def _binop(self, other: "Tile", kind: str) -> "Tile":
        result_type = Tile._get_type(self._height, self._width, self._layout)
        # EnumAttr syntax: #tiny_tile<ew_kind mul>
        kind_attr = Attribute.parse(f"#tiny_tile<ew_kind {kind}>")
        op = Operation.create(
            "tiny_tile.elementwise",
            results=[result_type],
            operands=[self._value, other._value],
            attributes={"kind": kind_attr}
        )
        return Tile._wrap(op.result, self._height, self._width, self._layout)

    def __add__(self, other): return self._binop(other, "add")
    def __sub__(self, other): return self._binop(other, "sub")
    def __mul__(self, other): return self._binop(other, "mul")
    def __truediv__(self, other): return self._binop(other, "div")

    @staticmethod
    def splat(value: float, height: int, width: int, layout: Layout) -> "Tile":
        """Create a tile filled with a constant value (tiny_tile.splat)."""
        tile_type = Tile._get_type(height, width, layout)
        val_attr = FloatAttr.get(F64Type.get(), value)
        op = Operation.create(
            "tiny_tile.splat",
            results=[tile_type],
            attributes={"splatValue": val_attr},
        )
        return Tile._wrap(op.result, height, width, layout)

    @staticmethod
    def zeros(height: int, width: int, layout: Layout) -> "Tile":
        """Create a zero-initialized tile."""
        return Tile.splat(0.0, height, width, layout)

    def reduce(self) -> Value:
        """Reduce tile to scalar f16 (tiny_tile.reduce)."""
        op = Operation.create(
            "tiny_tile.reduce",
            results=[F16Type.get()],
            operands=[self._value],
        )
        return op.result

    def layout_cast(self, new_layout: "Layout | None" = None) -> "Tile":
        """Cast tile to a different layout (or strip layout if None)."""
        result_type = Tile._get_type(self._height, self._width, new_layout)
        op = Operation.create(
            "tiny_tile.layout_cast",
            results=[result_type],
            operands=[self._value],
        )
        return Tile._wrap(op.result, self._height, self._width, new_layout)


def load_tile(ptr: Ptr, row: Index, col: Index, stride: Index,
              height: int, width: int, layout: Layout) -> Tile:
    """Load a tile from memory (tiny_tile.load).

    The layout specifies how elements are distributed across threads.
    Layout propagation will flow this through subsequent operations.
    """
    tile_type = Tile._get_type(height, width, layout)
    op = Operation.create(
        "tiny_tile.load",
        results=[tile_type],
        operands=[ptr._value, row._value, col._value, stride._value],
    )
    return Tile._wrap(op.result, height, width, layout)


def store_tile(tile: Tile, ptr: Ptr, row: Index, col: Index, stride: Index):
    """Store a tile to memory (tiny_tile.store)."""
    Operation.create(
        "tiny_tile.store",
        operands=[tile._value, ptr._value, row._value, col._value, stride._value],
    )


# --- Convenience functions ---

def _get_type_map():
    """Type map for ch3."""
    return {
        Ptr: (Ptr.get_type, Ptr),
        Index: (IndexType.get, Index),
    }


def print_ir(fn):
    """Print generated TinyTile dialect IR."""
    opt = TutorialOpt()
    with MLIRModule() as m:
        tile_ir = m.build_func_verified(fn, _get_type_map(), opt)
        print(tile_ir)
    return fn


def compile_and_print(fn):
    """Compile and print all lowering stages for ch3."""
    opt = TutorialOpt()

    with MLIRModule() as m:
        tile_ir = m.build_func_verified(fn, _get_type_map(), opt)

    print("=== TinyTile Dialect ===")
    print(tile_ir)

    layout_ir = opt.run(tile_ir, ["tiny-tile-layout-propagation"])
    print("=== After layout propagation ===")
    print(layout_ir)

    tiny_ir = opt.run(layout_ir, ["tiny-tile-to-tiny"])
    print("=== After tiny-tile-to-tiny ===")
    print(tiny_ir)

    arith_ir = opt.run(tiny_ir, ["tiny-to-arith"])
    print("=== After tiny-to-arith ===")
    print(arith_ir)

    return fn
