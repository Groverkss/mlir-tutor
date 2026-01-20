//===- TinyLoopDialect.cpp - TinyLoop dialect implementation ----*- C++ -*-===//
//
// Implements the TinyLoop dialect and its operations (Chapter 2).
//
// This file demonstrates:
// - Building operations with regions
// - Parsing and printing custom assembly formats
// - Verifying operation invariants for region-based ops
//
//===----------------------------------------------------------------------===//

#include "ch2-cpu-vector-dsl-loops/TinyLoopDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tiny_loop;

//===----------------------------------------------------------------------===//
// TinyLoop dialect implementation
//===----------------------------------------------------------------------===//

#include "ch2-cpu-vector-dsl-loops/TinyLoopOpsDialect.cpp.inc"

void TinyLoopDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ch2-cpu-vector-dsl-loops/TinyLoopOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AccumulateOp implementation
//===----------------------------------------------------------------------===//

/// Custom builder for AccumulateOp that sets up the region and block arguments.
///
/// The block arguments are structured as:
///   [induction_var: index, acc0: type0, acc1: type1, ...]
///
/// Where the induction variable is always present, and accumulator arguments
/// correspond to the initArgs.
void AccumulateOp::build(OpBuilder &builder, OperationState &result,
                         Value bound, Value step, ValueRange initArgs,
                         function_ref<void(OpBuilder &, Location, Value,
                                           ValueRange)> bodyBuilder) {
  result.addOperands({bound, step});
  result.addOperands(initArgs);
  result.addTypes(initArgs.getTypes());

  // Create the body region with a single block.
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = new Block();
  bodyRegion->push_back(bodyBlock);

  // Add the induction variable as the first block argument.
  // The induction variable is always of index type.
  bodyBlock->addArgument(builder.getIndexType(), result.location);

  // Add block arguments for each init arg (the accumulators).
  for (Value arg : initArgs) {
    bodyBlock->addArgument(arg.getType(), result.location);
  }

  // If a body builder was provided, use it to populate the body.
  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0),
                bodyBlock->getArguments().drop_front(1));
  }
}

/// Parse the custom assembly format for AccumulateOp.
///
/// Format:
///   tiny_loop.accumulate %bound step(%step) [init(%args...)] [-> types] { body }
///
/// Examples:
///   tiny_loop.accumulate %n step(%s) { ^bb0(%i: index): ... }
///   tiny_loop.accumulate %n step(%s) init(%x) -> index { ^bb0(%i: index, %a: index): ... }
ParseResult AccumulateOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse: %bound step(%step)
  OpAsmParser::UnresolvedOperand bound, step;
  if (parser.parseOperand(bound) ||
      parser.resolveOperand(bound, builder.getIndexType(), result.operands) ||
      parser.parseKeyword("step") || parser.parseLParen() ||
      parser.parseOperand(step) ||
      parser.resolveOperand(step, builder.getIndexType(), result.operands) ||
      parser.parseRParen())
    return failure();

  // Parse optional: init(%args...)
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<Type> initTypes;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    if (parser.parseLParen())
      return failure();

    // Parse comma-separated operands.
    if (parser.parseOptionalRParen()) {
      do {
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseOperand(operand))
          return failure();
        initOperands.push_back(operand);
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseRParen())
        return failure();
    }
  }

  // Parse optional: -> type or -> (type, type, ...)
  if (succeeded(parser.parseOptionalArrow())) {
    // Check if we have a parenthesized list of types.
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseTypeList(result.types) || parser.parseRParen())
        return failure();
    } else {
      // Single type without parentheses.
      Type type;
      if (parser.parseType(type))
        return failure();
      result.types.push_back(type);
    }
  }

  // Verify the number of init operands matches the number of result types.
  if (initOperands.size() != result.types.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "mismatch between number of init args (")
           << initOperands.size() << ") and result types ("
           << result.types.size() << ")";
  }

  // Resolve init operands with the result types.
  for (size_t i = 0; i < initOperands.size(); ++i) {
    if (parser.resolveOperand(initOperands[i], result.types[i],
                              result.operands))
      return failure();
  }

  // Parse optional attributes (using keyword format to avoid confusion with region).
  // This expects "attributes { ... }" rather than just "{ ... }".
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the region.
  // The region has block arguments: [induction_var: index, acc0, acc1, ...]
  Region *body = result.addRegion();

  // We need to know the block argument types before parsing.
  // Block args: index (for induction var) + result types (for accumulators).
  SmallVector<OpAsmParser::Argument> regionArgs;

  // Induction variable argument.
  OpAsmParser::Argument ivArg;
  ivArg.type = builder.getIndexType();
  regionArgs.push_back(ivArg);

  // Accumulator arguments (same types as results).
  for (Type type : result.types) {
    OpAsmParser::Argument accArg;
    accArg.type = type;
    regionArgs.push_back(accArg);
  }

  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Ensure the body has a terminator.
  AccumulateOp::ensureTerminator(*body, builder, result.location);

  return success();
}

/// Print the custom assembly format for AccumulateOp.
void AccumulateOp::print(OpAsmPrinter &p) {
  p << " " << getBound() << " step(" << getStep() << ")";

  // Print init args if present.
  if (!getInitArgs().empty()) {
    p << " init(";
    llvm::interleaveComma(getInitArgs(), p);
    p << ")";
  }

  // Print result types if present.
  if (!getResultTypes().empty()) {
    p << " -> ";
    if (getResultTypes().size() > 1)
      p << "(";
    llvm::interleaveComma(getResultTypes(), p);
    if (getResultTypes().size() > 1)
      p << ")";
  }

  // Print attributes if any (using keyword format).
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());

  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/!getInitArgs().empty());
}

/// Verify the AccumulateOp invariants.
LogicalResult AccumulateOp::verify() {
  // Check that the region has exactly one block.
  if (getRegion().getBlocks().size() != 1)
    return emitOpError("expected exactly one block in the region");

  Block *body = getBody();

  // Check that we have at least one block argument (induction variable).
  if (body->getNumArguments() < 1)
    return emitOpError("expected at least one block argument (induction var)");

  // Check that the first block argument is index type.
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError("induction variable must be of index type");

  // Check that the number of init args matches the number of results.
  if (getInitArgs().size() != getNumResults())
    return emitOpError("mismatch between number of init args (")
           << getInitArgs().size() << ") and results (" << getNumResults()
           << ")";

  // Check that block arguments (after IV) match init arg types.
  size_t expectedBlockArgs = 1 + getInitArgs().size();
  if (body->getNumArguments() != expectedBlockArgs)
    return emitOpError("expected ")
           << expectedBlockArgs << " block arguments, got "
           << body->getNumArguments();

  for (auto [blockArg, initArg] :
       llvm::zip(getRegionIterArgs(), getInitArgs())) {
    if (blockArg.getType() != initArg.getType())
      return emitOpError("block argument type mismatch: expected ")
             << initArg.getType() << ", got " << blockArg.getType();
  }

  // Check that the yield has the right number and types of operands.
  auto yield = cast<YieldOp>(body->getTerminator());
  if (yield.getResults().size() != getNumResults())
    return emitOpError("yield has ")
           << yield.getResults().size() << " operands, but loop has "
           << getNumResults() << " results";

  for (auto [yieldVal, resultType] :
       llvm::zip(yield.getResults(), getResultTypes())) {
    if (yieldVal.getType() != resultType)
      return emitOpError("yield operand type mismatch: expected ")
             << resultType << ", got " << yieldVal.getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ch2-cpu-vector-dsl-loops/TinyLoopOps.cpp.inc"
