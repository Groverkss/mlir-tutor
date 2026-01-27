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
