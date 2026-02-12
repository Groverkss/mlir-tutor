"""DSL wrapper types for Tiny dialect."""

from mlir.ir import (
    Value,
    Type,
    Operation,
    IndexType,
    VectorType,
    F16Type,
    IntegerAttr,
    DenseElementsAttr,
    register_value_caster,
)
import numpy as np


@register_value_caster(IndexType.static_typeid)
class Index(Value):
    """Wrapper for index-typed SSA values."""

    @staticmethod
    def constant(val: int) -> "Index":
        """Create a constant index value via tiny.constant."""
        idx_type = IndexType.get()
        attr = IntegerAttr.get(idx_type, val)
        op = Operation.create(
            "tiny.constant",
            results=[idx_type],
            attributes={"value": attr},
        )
        return op.result

    def _binop(self, other: "Index", op_name: str) -> "Index":
        op = Operation.create(
            f"tiny.{op_name}",
            results=[IndexType.get()],
            operands=[self._value, other._value],
        )
        return op.result

    def __add__(self, other):
        return self._binop(other, "addi")

    def __sub__(self, other):
        return self._binop(other, "subi")

    def __mul__(self, other):
        return self._binop(other, "muli")

    def __floordiv__(self, other):
        return self._binop(other, "divi")


class F16Vector(Value):
    """Wrapper for vector<Nxf16> SSA values."""

    @staticmethod
    def constant(vals: list[float], size: int = None) -> "F16Vector":
        """Create a constant f16 vector via tiny.constant."""
        n = size or len(vals)
        vec_type = VectorType.get([n], F16Type.get())
        data = np.array(vals, dtype=np.float16)
        attr = DenseElementsAttr.get(data, type=vec_type)
        op = Operation.create(
            "tiny.constant",
            results=[vec_type],
            attributes={"value": attr},
        )
        return op.result

    def _binop(self, other: "F16Vector", op_name: str) -> "F16Vector":
        op = Operation.create(
            f"tiny.{op_name}",
            results=[self._value.type],
            operands=[self._value, other._value],
        )
        return op.result

    def __add__(self, other):
        return self._binop(other, "addf")

    def __sub__(self, other):
        return self._binop(other, "subf")

    def __mul__(self, other):
        return self._binop(other, "mulf")

    def __truediv__(self, other):
        return self._binop(other, "divf")

    def sum(self) -> "F16Vector":
        """Reduce to vector<1xf16> via tiny.sum."""
        result_type = VectorType.get([1], F16Type.get())
        op = Operation.create(
            "tiny.sum",
            results=[result_type],
            operands=[self._value],
        )
        return op.result


@register_value_caster(VectorType.static_typeid)
def maybe_wrap_vector(val: Value):
    if isinstance(val.type.element_type, F16Type):
        return F16Vector(val)
    return val


class Ptr:
    """Wrapper for !tiny.ptr SSA values."""

    _value: Value

    @staticmethod
    def _wrap(value: Value) -> "Ptr":
        p = Ptr()
        p._value = value
        return p

    @staticmethod
    def get_type() -> Type:
        """Get the !tiny.ptr type."""
        return Type.parse("!tiny.ptr")

    def load(self, offset: Index, num_elements: int) -> F16Vector:
        """Load vector<Nxf16> from pointer at offset."""
        vec_type = VectorType.get([num_elements], F16Type.get())
        op = Operation.create(
            "tiny.load",
            results=[vec_type],
            operands=[self._value, offset._value],
        )
        return F16Vector._wrap(op.result)

    def store(self, offset: Index, vec: F16Vector) -> None:
        """Store vector<Nxf16> to pointer at offset."""
        Operation.create(
            "tiny.store",
            results=[],
            operands=[vec._value, self._value, offset._value],
        )


# --- Convenience functions ---

from ..compiler import MLIRModule, TutorialOpt


def _get_type_map():
    """Type map for ch1 tiny dialect."""
    return {
        Ptr: (Ptr.get_type, Ptr),
        Index: (IndexType.get, Index),
    }


def print_ir(fn):
    """Print generated Tiny dialect IR (verified and pretty-printed)."""
    opt = TutorialOpt()
    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)
        print(tiny_ir)
    return fn


def compile_and_print(fn):
    """Compile and print all lowering stages."""
    opt = TutorialOpt()

    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)

    print("=== Tiny Dialect ===")
    print(tiny_ir)

    arith_ir = opt.run(tiny_ir, ["tiny-to-arith", "canonicalize", "cse"])
    print("=== After tiny-to-arith ===")
    print(arith_ir)

    llvm_ir = opt.run(
        tiny_ir,
        ["tiny-to-arith", "canonicalize", "cse", "tiny-to-llvm", "convert-to-llvm"],
    )
    print("=== LLVM Dialect ===")
    print(llvm_ir)

    return llvm_ir
