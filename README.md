# MLIR Wheel Tutorial

This repository is a small showcase for building an out-of-tree MLIR dialect
against the packaged `mlir-wheel` release instead of a local LLVM/MLIR source
tree.

The tutorial dialect lives in `tutorial/list`. Its TableGen file defines the
entire dialect surface:

- `!list.list<T>`: a parameterized list type.
- `list.empty`: creates an empty list.
- `list.append`: appends one value to a list.
- `list.concat`: concatenates two lists.
- `list.length`: returns a list length as `index`.

The C++ files in the same directory are the minimal wrapper needed to register
the generated dialect, type, and operation definitions with MLIR.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .env
source .env/bin/activate
```

Install the Python packages:

```bash
pip install -r requirements.txt
```

Check that the MLIR wheel is available:

```bash
python -m mlir_wheel --root-dir
```

## Build

Configure CMake against the MLIR wheel:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) \
  -DLLVM_EXTERNAL_LIT=$(which lit)
```

Build `tutorial-opt`:

```bash
cmake --build build --target tutorial-opt
```

Run the list dialect roundtrip tests:

```bash
cmake --build build --target check-tutorial-list
```

Run all tutorial tests:

```bash
cmake --build build --target check-tutorial
```

## Try It

Roundtrip a list dialect example through `tutorial-opt`:

```bash
build/tutorial/tutorial-opt tutorial/list/test/roundtrip.mlir
```

The test file uses `FileCheck` to verify that the custom list type and list
operations parse and print correctly.
