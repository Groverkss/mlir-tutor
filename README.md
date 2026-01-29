## Installation Instructions

1. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Check if the installation was successful:

```bash
python3 -m mlir_wheel --root-dir
```

4. Configure CMake:

```bash
mkdir build
cd build
cmake ../ -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir)
```

5. Build:

```bash
make -j
```

## Running Python Examples

1. Install the Python DSL package (editable mode):

```bash
pip install -e python/
```

2. Set the `TUTORIAL_OPT` environment variable to point to the built `tutorial-opt` binary:

```bash
export TUTORIAL_OPT=$PWD/build/tutorial/tutorial-opt
```

3. Run an example:

```bash
# Chapter 1: Vectorized square
python examples/ch1/ch1_square.py

# Chapter 2: Matrix multiplication with loops
python examples/ch2/ch2_matmul.py

# Chapter 3: GPU tile operations
python examples/ch3/ch3_tile_elemwise.py
```

The examples will print the generated MLIR at each lowering stage.
