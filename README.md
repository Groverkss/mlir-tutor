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
cmake ../ -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) -DLLVM_EXTERNAL_LIT=$(which lit)
```

5. Build:

```bash
make -j
```

## Installing Python Package

1. Install the Python DSL package (editable mode):

```bash
# From project root directory (not the build directory).
pip install -e python/
```

2. Set the `TUTORIAL_OPT` environment variable to point to the built `tutorial-opt` binary:

```bash
export TUTORIAL_OPT=$PWD/build/tutorial/tutorial-opt
```

3. Run an example to verify:

```bash
python tutorial/ch1-cpu-vector-dsl/square.py
```

The examples will print the generated MLIR at each lowering stage.

## Doing tutorial exercises

Each tutorial chapter has a README.md file explaining how to start with the exercise and some basic explanation of the exercise. For example, check out tutorial/ch1-cpu-vector-dsl/README.md
