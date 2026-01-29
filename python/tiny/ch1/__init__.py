"""Chapter 1: Tiny dialect DSL."""

from .dsl import Index, F16Vector, Ptr
from .dsl import print_ir, compile_and_print

__all__ = [
    "Index", "F16Vector", "Ptr",
    "print_ir", "compile_and_print",
]
