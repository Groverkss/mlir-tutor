# Ch2: CPU Vector DSL with Loops

In this chapter, we will extend the language defined in ch1 with a new loop
operation.

## Language

We can use the accumulate operation to define loops:

```python
def matmul(a: Ptr, b: Ptr, c: Ptr, M: Index, N: Index, K: Index):
    vec_size = 16
    one = Index.constant(1)
    vstep = Index.constant(vec_size)

    @accumulate(M, one)
    def _(i: Index):
        @accumulate(N, one)
        def _(j: Index):
            # Initialize accumulator vector
            acc_init = F16Vector.constant([0.0])

            @accumulate(K, vstep, inits=[acc_init])
            def _(k: Index, acc: F16Vector):
                # Load a[i, k:k+16] and b[j, k:k+16]
                a_idx = i * K + k
                b_idx = j * K + k
                a_vec = a.load(a_idx, num_elements=vec_size)
                b_vec = b.load(b_idx, num_elements=vec_size)
                # Accumulate element-wise product
                return acc + (a_vec * b_vec).sum()

            # Sum the accumulator and store result
            result = _[0]
            c_idx = i * N + j
            c.store(c_idx, result)
```

Note that the accumulate operation takes a size and a step, and produces a loop
equivalent to pythons: `for i in range(0, size, step)`.

The operation takes a body, which is passed the induction variable as it's
first argument for the current iteration.

Since our language is SSA based, the loop also supports loop carried variables.
The initial value of the loop carried variables can be passed as list of
`inits`. The body is passed the current value of the loop carried variables.
The body must return the updated value of the loop carried variables.

## Lowering

We implement 1 pass in this chapter:

- TinyLoopToSCF:

We lower our tiny_loop.accumulate operation to MLIR's upstream scf dialect
operations: https://mlir.llvm.org/docs/Dialects/SCFDialect/ . Specifically, the
scf.for operation:
https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop

The pattern to implement has pseudo code of what needs to be implemented.

## Playing around with the language

Switch to the `main` branch. Make sure you have the python package installed
as mentioned in ../../README.md

Open `matmul.py` and have a look at the python dsl code. Run the file with
`python3 matmul.py` to see the language lowering to LLVM.

You can use this as a playground to play around with the language and see what
code is produced during the lowerings. This should get you a better
understanding of what the operations look like and what the lowerings look
like.

Try writing a dot product `sum(a * b)` in the language to get more familiar
with it.

## Exercise Implementations

Switch to `ch2-excercise` branch to access the exercises for ch2.

### Exercise 1: TinyLoopToSCF Lowering Pass

Open the TinyLoopToSCF.cpp file. You will notice there is already a pass
skeleton defined for you. You need to implement a pattern to rewrite
the `tiny_loop.accumulate` operation to `scf.for` operation.

There is pseudo-code to help you understand what should be emitted.

Once done, try testing with `test/lower-to-scf.mlir` file, which should verify
if the lowering works as expected. Note that if there is an error the flag
`--mlir-print-ir-after-failure` helps see what the ir generated looks like if
it failed, so you can debug what code you are producing.

### Challenge Exercise (Optional)

Try adding a `tiny_loop.if_zero` operation and lowering it to `scf.if` operation.
Look at the `scf.if` operation for motivation.

The operation should take the `then` region if the index value passed to it is
0, otherwise it should take the `else` region.
