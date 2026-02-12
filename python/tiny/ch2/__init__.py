"""Chapter 2: TinyLoop dialect DSL."""

from ..ch1 import Index, F16Vector, Ptr
from .dsl import accumulate, print_ir, compile_and_print

__all__ = [
    "Index",
    "F16Vector",
    "Ptr",
    "accumulate",
    "print_ir",
    "compile_and_print",
]
