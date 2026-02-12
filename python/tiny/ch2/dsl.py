"""DSL for TinyLoop dialect (Chapter 2)."""

from mlir.ir import (
    Value,
    Block,
    Operation,
    InsertionPoint,
    IndexType,
)

# Re-export ch1 types
from ..ch1.dsl import Index, F16Vector, Ptr
from ..compiler import MLIRModule, TutorialOpt


def _wrap_value(value: Value, template):
    """Wrap MLIR value using the same type as template."""
    # Try passing template for types that need it (e.g., Tile)
    if isinstance(value, (F16Vector, Index)):
        return value

    try:
        return type(template)._wrap(value, template=template)
    except TypeError:
        return type(template)._wrap(value)


def accumulate(bound: Index, step: Index, inits: list = None):
    """Create a tiny_loop.accumulate loop.

    Usage:
        @accumulate(n, step)
        def loop_body(i: Index):
            # side-effect only loop
            ...

        @accumulate(n, step, inits=[acc])
        def loop_body(i: Index, acc: F16Vector):
            # loop with accumulator
            ...
            return new_acc
        result = loop_body  # captures the result
    """
    inits = inits or []

    def decorator(body_fn):
        # Get init value types and MLIR values
        init_values = [
            v if isinstance(v, (Index, F16Vector)) else v._value for v in inits
        ]
        init_types = [
            v.type if isinstance(v, (Index, F16Vector)) else v._value.type
            for v in inits
        ]
        result_types = init_types  # Results match init types

        # Create the accumulate op with one region
        op = Operation.create(
            "tiny_loop.accumulate",
            results=result_types,
            operands=[bound, step] + init_values,
            regions=1,  # One region for the body
        )

        # Set up the block with arguments: (index, *iter_args)
        region = op.regions[0]
        block_arg_types = [IndexType.get()] + init_types
        block = Block.create_at_start(region, block_arg_types)

        # Execute body with wrapped arguments
        with InsertionPoint(block):
            iv = block.arguments[0]
            iter_args = [
                _wrap_value(block.arguments[i + 1], inits[i]) for i in range(len(inits))
            ]

            # Call user's body function
            if inits:
                results = body_fn(iv, *iter_args)
                if not isinstance(results, (list, tuple)):
                    results = [results]
                yield_values = [
                    r if isinstance(r, (Index, F16Vector)) else r._value
                    for r in results
                ]
            else:
                body_fn(iv)
                yield_values = []

            # Create tiny_loop.yield
            Operation.create("tiny_loop.yield", operands=yield_values)

        # Wrap and return results
        if result_types:
            return [
                _wrap_value(op.results[i], inits[i]) for i in range(len(result_types))
            ]
        return None

    return decorator


# --- Convenience functions ---


def _get_type_map():
    """Type map for ch2 (same as ch1)."""
    return {
        Ptr: (Ptr.get_type, Ptr),
        Index: (IndexType.get, Index),
    }


def print_ir(fn):
    """Print generated Tiny+TinyLoop dialect IR."""
    opt = TutorialOpt()
    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)
        print(tiny_ir)
    return fn


def compile_and_print(fn):
    """Compile and print all lowering stages for ch2."""
    opt = TutorialOpt()

    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)

    print("=== Tiny + TinyLoop Dialect ===")
    print(tiny_ir)

    # Lower tiny_loop.accumulate -> scf.for
    scf_ir = opt.run(tiny_ir, ["tiny-loop-to-scf"])
    print("=== After tiny-loop-to-scf ===")
    print(scf_ir)

    # Lower tiny ops -> arith
    arith_ir = opt.run(scf_ir, ["tiny-to-arith", "canonicalize", "cse"])
    print("=== After tiny-to-arith ===")
    print(arith_ir)

    # Lower to LLVM
    llvm_ir = opt.run(
        arith_ir, ["tiny-to-llvm", "convert-scf-to-cf", "convert-to-llvm"]
    )
    print("=== LLVM Dialect ===")
    print(llvm_ir)

    return fn
