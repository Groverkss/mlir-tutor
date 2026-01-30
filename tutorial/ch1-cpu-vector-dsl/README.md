# Ch1: CPU Vector DSL

In this chapter, we will define a simple language that contains operations
for computing floating point arithmetic on fp16 vectors.

## Operations

We will define the following operations:

Floating Point Vector Operations:

- tiny.constant: Create a vector filled with a constant fp16 value.
- tiny.addf: Element-wise addition of two fp16 vectors.
- tiny.subf: Element-wise subtraction of two fp16 vectors.
- tiny.mulf: Element-wise multiplication of two fp16 vectors.
- tiny.divf : Element-wise division of two fp16 vectors.
- tiny.sum: Sum all elements of an fp16 vector to produce a single fp16 value.

Indexing Operations:

- tiny.addi: Element-wise addition of two index integers.
- tiny.subi: Element-wise subtraction of two index integers.
- tiny.muli: Element-wise multiplication of two index integers.
- tiny.divi: Element-wise division of two index integers.

Memory Operations:

- tiny.load: Load an fp16 vector from memory given a base pointer and an index.
- tiny.store: Store an fp16 vector to memory given a base pointer and an index.

Types:

- tiny.ptr: Pointer to fp16 memory, with fp16 offsets

## Language

With this language we can represent some simple programs:

```python
def square(a: Ptr, result: Ptr, offset: Index):
    x = a.load(offset, num_elements=16)
    x_squared = x * x
    result.store(offset, x_squared)
```

## Lowering

We implement 2 passes in this chapter:

- TinyToArith:

We lower the tiny arithmetic (fp16 and index) to MLIR's upstream arith dialect
operations: https://mlir.llvm.org/docs/Dialects/ArithOps/ (It's always useful
to google "dialect name mlir" to find the docs and see what is available and
what their semantics are).

We lower:

- tiny.constant -> arith.constant
- tiny.addf -> arith.addf
- tiny.subf -> arith.subf
- tiny.mulf -> arith.mulf
- tiny.divf -> arith.divf
- tiny.sum -> arith.addf + arith.reduction
- tiny.addi -> arith.addi
- tiny.subi -> arith.subi
- tiny.muli -> arith.muli
- tiny.divi -> arith.divi

These are simple 1-1 (or 1-2) mapping of operations, which will be done
using MLIR's rewrite system. This is one of the simplest ways of rewriting
operations in MLIR.

- TinyToLLVM:

We lower the tiny memory operations to LLVM dialect operations:
https://mlir.llvm.org/docs/Dialects/LLVM/

We lower:

- tiny.load -> llvm.getelementptr + llvm.load
- tiny.store -> llvm.getelementptr + llvm.store

## Fully lowering to LLVM

With these passes, we can fully lower our tiny language to MLIR, by reusing
arith -> llvm conversions. This is one of MLIR's strengths: reusing existing
infrastructure whenever possible.

## Playing around with the language

Switch to the `main` branch. Make sure you have the python package installed
as mentioned in ../../README.md

Open `square.py` and have a look at the python dsl code. Run the file with
`python3 square.py` to see the language lowering to LLVM.

You can use this as a playground to play around with the language and see what
code is produced during the lowerings. This should get you a better
understanding of what the operations look like and what the lowerings look
like.

Try writing an elementwise add in the language to get more familiar with it.

## Exercise Implementations

Switch to `ch1-excercise` branch to access the exercises for ch1.

### Exercise 1: Op Implementations

Open the TinyDialect.td file. You will notice there are some dialect / type /
operation definitions that already exist. The following things are already
implemented:

- tiny dialect definition (we will learn about defining dialects later, for
  now, it's just a namespace for operations)

- tiny.ptr type definition (we will learn more custom types later)

- tiny.constant

- tiny.addf
- tiny.sum

- tiny.addi

- tiny.load

The task for this exercise is to implement the operation definition of the
remaining operations. You should look at how other operations are defined, and
simply reuse the same pattern (FYI: The author has been working on MLIR for
over 5 years now and still copies tablegen definitions from elsewhere).

Once you are done, try testing with `test/basic-ops.mlir` file, which should
verify if the operations definitions work as expected.

### Exercise 2: TinyToArith Lowering Pass

Open the TinyToArith.cpp file. You will notice there is already a pass skeleton
defined for you, with the following patterns already defined (for motivation):

- tiny.constant -> arith.constant
- tiny.addf -> arith.addf
- tiny.sum -> arith.addf + arith.reduction
- tiny.addi -> arith.addi

Your task is to implement the lowering patterns for the remaining operations:

- tiny.subf -> arith.subf
- tiny.mulf -> arith.mulf
- tiny.divf -> arith.divf
- tiny.subi -> arith.subi
- tiny.muli -> arith.muli
- tiny.divi -> arith.divi

Once done, you can try testing with `test/lower-to-arith.mlir`  file, which
should verify if the lowering works as expected.

### Exercise 3: TinyToLLVM Lowering Pass

Open the TinyToLLVM.cpp file, you will notice there is already a pass skeleton
defined for you, with the following pattern already defined (for motivation):

- tiny.load -> llvm.getelementptr + llvm.load

Your task is to implement the lowering pattern for the remaining operation:

- tiny.store -> llvm.getelementptr + llvm.store

Once done, you can try testing with `test/lower-to-llvm.mlir` file, which
should verify if the lowering works as expected.

Note that this file uses more complex infrastructure for conversion, which
also converts types. We will learn about this more later, but for now,
you can add a TypeConverter in rewrite pattern that allows type conversions.

Once done, you can try testing with `test/lower-to-llvm.mlir` file, which
should verify if the lowering works as expected.

### Challenge Exercise (Optional)

Write a matmul computation of M=64,N=64,K=64 in the python dsl, using only
vectors of width 8.
Hint: Use python loops directly
