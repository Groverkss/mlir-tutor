//===- TinyLoopDialect.td - Tiny Loop dialect definition ---*- tablegen -*-===//
//
// Defines the TinyLoop dialect for the MLIR tutorial (Chapter 2).
// This dialect introduces MLIR Regions and Basic Blocks through a simple
// accumulate loop operation.
//
//===----------------------------------------------------------------------===//

#ifndef TINY_LOOP_DIALECT
#define TINY_LOOP_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// TinyLoop dialect definition
//===----------------------------------------------------------------------===//

def TinyLoop_Dialect : Dialect {
  let name = "tiny_loop";
  let summary = "A simple loop dialect for the MLIR tutorial (Chapter 2).";
  let description = [{
    The tiny_loop dialect provides a simple loop operation to model loops.
  }];
  let cppNamespace = "::mlir::tiny_loop";
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class TinyLoop_Op<string mnemonic, list<Trait> traits = []> :
    Op<TinyLoop_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Accumulate operation
//===----------------------------------------------------------------------===//

def TinyLoop_AccumulateOp : TinyLoop_Op<"accumulate", [
    // The region has exactly one block.
    SingleBlock,
    // The block is implicitly terminated by tiny_loop.yield.
    SingleBlockImplicitTerminator<"YieldOp">,
    // The loop may have memory effects through its body.
    RecursiveMemoryEffects
  ]> {
  let summary = "Accumulate loop from 0 to N.";
  let description = [{
    The `tiny_loop.accumulate` operation iterates from 0 to N (exclusive)
    with a given step. The loop body is a region with a single block.

    The block's first argument is always the induction variable (index type).
    If `init` values are provided, they become additional block arguments
    and the loop produces results.

    Examples:

    ```mlir
    // Side-effect only loop (no accumulation)
    tiny_loop.accumulate %n step(%step) {
    ^bb0(%i: index):
      // loop body
      tiny_loop.yield
    }

    // Loop with accumulation
    %result = tiny_loop.accumulate %n step(%step) init(%init) -> index {
    ^bb0(%i: index, %acc: index):
      %next = arith.addi %acc, %i : index
      tiny_loop.yield %next : index
    }

    // Multiple accumulators
    %a, %b = tiny_loop.accumulate %n step(%step) init(%x, %y) -> (i32, f32) {
    ^bb0(%i: index, %acc_a: i32, %acc_b: f32):
      // ...
      tiny_loop.yield %new_a, %new_b : i32, f32
    }
    ```
  }];

  let arguments = (ins
    Index:$bound,
    Index:$step,
    Variadic<AnyType>:$initArgs
  );

  let results = (outs Variadic<AnyType>:$results);

  // SizedRegion<1> means the region has exactly one block.
  let regions = (region SizedRegion<1>:$region);

  // Custom assembly format for the syntax:
  // tiny_loop.accumulate %N step(%step) [init(%args)] [-> types] { body }
  let hasCustomAssemblyFormat = 1;

  // Custom builder to set up the block arguments correctly.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$bound, "Value":$step,
      CArg<"ValueRange", "{}">:$initArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    /// Get the induction variable (first block argument).
    Value getInductionVar() {
      return getBody()->getArgument(0);
    }

    /// Get the region iteration arguments (block arguments after induction var).
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front(1);
    }

    /// Get the body block.
    Block *getBody() { return &getRegion().front(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Yield operation
//===----------------------------------------------------------------------===//

def TinyLoop_YieldOp : TinyLoop_Op<"yield", [
    Pure,
    Terminator,
    HasParent<"AccumulateOp">
  ]> {
  let summary = "Yield values from a tiny_loop.accumulate body.";
  let description = [{
    The `tiny_loop.yield` operation terminates the body of a
    `tiny_loop.accumulate` loop. If the loop has `init` args, the yield
    must provide the values for the next iteration (or final results).

    The number and types of yielded values must match the `init` args.

    Examples:

    ```mlir
    // For loops without init
    tiny_loop.yield

    // For loops with init
    tiny_loop.yield %next : index

    // For multiple accumulators
    tiny_loop.yield %a, %b : i32, f32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>
  ];
}

#endif // TINY_LOOP_DIALECT
