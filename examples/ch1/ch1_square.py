"""Example: Vectorized square function using ch1 tiny dialect."""

from tiny.ch1 import Ptr, Index, compile_and_print

@compile_and_print
def square(a: Ptr, result: Ptr, offset: Index):
    """Square a vector of 16 f16 elements."""
    x = a.load(offset, num_elements=16)
    x_squared = x * x
    result.store(offset, x_squared)
