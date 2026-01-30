# Ch4: Linalg and Transform Dialect

In this chapter, we explore two MLIR dialects:

1. **Linalg Dialect**: A high-level abstraction for linear algebra operations
2. **Transform Dialect**: A Halide-like IR, which allows us to do
   transformations on other IR

Unlike previous chapters where we defined custom dialects, here we learn to use
existing MLIR infrastructure and extend the transform dialect with custom
operations.

## Linalg Dialect

The linalg dialect provides structured operations for linear algebra:

### linalg.generic

The fundamental building block of the linalg dialect, linalg.generic,
describes a perfectly nested loop.

```mlir
#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>

%result = linalg.generic {
  indexing_maps = [#map_A, #map_B, #map_C],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
  outs(%C : tensor<64x64xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %prod = arith.mulf %a, %b : f32
    %sum = arith.addf %c, %prod : f32
    linalg.yield %sum : f32
} -> tensor<64x64xf32>
```

The body of the linalg.generic defines the body of the loop. The inputs to
the body are elements accessed from the inputs/outputs at that iteration
of the loop nest.

The `indexing_maps` define how the inputs/outputs are accessed as a function
of the loop nest's iteration space. In the above example,

at the iteration space point (m=2, n=3, k=4), the body is passed the values:

```mlir
%a = %A[2, 4]
%b = %B[4, 3]
%c = %C[2, 3]
```

The `iterator_types` define the type of each loop in the loop nest.

## Transform Dialect

Note that we only give a simple introduction to Transform dialect. The
transform dialect has an excelent tutorial upstream:
https://mlir.llvm.org/docs/Tutorials/transform/

The transform dialect lets you write transformation scripts that operate on IR:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Find operations to transform by matching by name.
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Apply tiling to the matched operation.
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul
        tile_sizes [16, 16, 8]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

Run with: `tutorial-opt --transform-interpreter input.mlir`

## Examples

The `examples/` directory contains progressive examples showing the linalg
transformation pipeline:

### Example 1: Linalg to Loops

Shows how `linalg.generic` operations conceptually map to nested loops:

```bash
tutorial-opt --convert-linalg-to-loops examples/01-linalg-to-loops.mlir
```

### Example 2: Tiling

Shows how to tile a single linalg operation using the transform dialect:

```bash
tutorial-opt --transform-interpreter examples/02-tile-single-op.mlir
```

### Example 3: Tile and Fuse

Demonstrates the key pattern: **tile the consumer first, then fuse the producer**.

Pipeline:
1. Tile add (consumer) with `scf.forall` for parallel execution
2. Fuse matmul (producer) into the forall
3. Fuse fill (matmul's producer) into the forall
4. Tile the reduction dimension with `scf.for`

```bash
tutorial-opt --transform-interpreter examples/03-tile-and-fuse.mlir
```

### Example 4: Vectorization

Extends example 3 by adding vectorization after tiling and fusion:

```bash
tutorial-opt --transform-interpreter examples/04-vectorize.mlir
```

### Example 5: Full Lowering

Shows the complete pipeline from tensors to LLVM IR:

```bash
tutorial-opt --transform-interpreter \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --convert-vector-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    examples/05-bufferize-to-llvm.mlir
```

## Exercises

The `exercises/` directory contains exercises to practice transform dialect usage:

### Exercise 1: L2/L1 Tiling Without Fusion

Open the file: `exercises/ex1-l2-l1-no-fusion.mlir` and implement two-level
cache tiling for matmul and add, tiled separately:

1. Tile matmul with L2 sizes [64, 64, 32] using `scf.for`
2. Tile matmul again with L1 sizes [16, 16, 8] using `scf.for`
3. Tile add with L2 sizes [64, 64] using `scf.for`
4. Tile add again with L1 sizes [16, 16] using `scf.for`

Take any L2/L1 sizes and any problem shape.

### Exercise 2: L2/L1 Tiling With Fusion

Open the file: `exercises/ex2-l2-l1-with-fusion.mlir` and implement two-level
tiling with fusion.

1. Tile add (consumer) with L2 sizes using `scf.forall`
2. Fuse matmul (producer) into the forall
3. Fuse fill into the forall
4. Tile reduction dimension of matmul
5. Apply L1 tiling to matmul and add separately

Take any L2/L1 sizes and any problem shape.

## Exercise 3: Custom Transform Operation `transform.tiny.tile_to_l2_l1`

Implement a transform op that performs two-level tiling in one step:

```mlir
%matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
%tiled = transform.tiny.tile_to_l2_l1 %matmul
    l2 [64, 64, 32] l1 [16, 16, 8]
    : (!transform.any_op) -> !transform.any_op
```

Open `TinyTransformOps.td` and add the definition of
`transform.tiny.tile_to_l2_l1` op.

Open `TinyTransformOps.cpp` and find `TinyTileToL2L1Op::applyToOne`.
Follow the TODO comments to implement:

1. Cast target to `TilingInterface`
2. Create `SCFTilingOptions` for L2 tiling
3. Call `scf::tileUsingSCF` for L2
4. Repeat for L1 on the tiled result

## Testing

Run the tests from the build directory:

```bash
make check-tutorial-ch4
```
