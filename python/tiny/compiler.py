"""Common compiler infrastructure for the tutorial DSL."""

from mlir.ir import Context, Location, Module, InsertionPoint
import mlir.dialects.func as func_d
import subprocess
import os
import inspect
from typing import Callable


def _get_tutorial_opt() -> str:
    """Get tutorial-opt binary from TUTORIAL_OPT environment variable."""
    path = os.environ.get("TUTORIAL_OPT")
    if not path:
        raise RuntimeError(
            "TUTORIAL_OPT environment variable not set.\n"
            "Please set it to the path of tutorial-opt binary:\n"
            "  export TUTORIAL_OPT=/path/to/build/tutorial/tutorial-opt"
        )
    return path


class TutorialOpt:
    """Wrapper around tutorial-opt for running passes."""

    def __init__(self, binary_path: str = None):
        self.binary = (binary_path or _get_tutorial_opt()).strip()

    def run(self, ir: str, passes: list[str]) -> str:
        """Run passes on IR string, return transformed IR."""
        args = [self.binary] + [f"--{p}" for p in passes]
        result = subprocess.run(args, input=ir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"tutorial-opt failed:\n{result.stderr}")
        return result.stdout


class MLIRModule:
    """Context manager for building MLIR modules with unregistered dialects."""

    def __init__(self):
        self.ctx = None
        self.loc = None
        self.module = None

    def __enter__(self):
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.ctx.__enter__()
        self.loc = Location.unknown()
        self.loc.__enter__()
        return self

    def __exit__(self, *args):
        self.loc.__exit__(*args)
        self.ctx.__exit__(*args)

    def build_func(self, fn: Callable, type_map: dict) -> Module:
        """Build MLIR module from a Python function.

        Args:
            fn: Function to compile (uses type annotations for args)
            type_map: Maps annotation types to (mlir_type, wrapper_class) tuples
        """
        # prevent circular import
        from tiny.ch1 import Ptr

        sig = inspect.signature(fn)
        self.module = Module.create()

        with InsertionPoint(self.module.body):
            # Build input types from annotations
            input_types = []
            for param in sig.parameters.values():
                if param.annotation not in type_map:
                    raise ValueError(f"Unsupported type: {param.annotation}")
                mlir_type, _ = type_map[param.annotation]
                input_types.append(mlir_type() if callable(mlir_type) else mlir_type)

            # Create func.func
            func_op = func_d.FuncOp(fn.__name__, (input_types, []))

            with InsertionPoint(func_op.add_entry_block()):
                # Wrap block arguments in DSL types
                args = []
                for i, param in enumerate(sig.parameters.values()):
                    _, wrapper_cls = type_map[param.annotation]
                    if wrapper_cls is Ptr:
                        args.append(wrapper_cls._wrap(func_op.arguments[i]))
                    else:
                        args.append(func_op.arguments[i])

                # Execute user function body
                fn(*args)

                # Add return
                func_d.return_([])

        return self.module

    def build_func_verified(
        self, fn: Callable, type_map: dict, opt: "TutorialOpt"
    ) -> str:
        """Build and verify module, return pretty-printed IR.

        Runs the generated IR through tutorial-opt to verify it's valid
        and get pretty-printed output using the dialect's assembly format.
        """
        self.build_func(fn, type_map)
        raw_ir = str(self.module)
        # Round-trip through tutorial-opt to verify and pretty-print
        return opt.run(raw_ir, [])
