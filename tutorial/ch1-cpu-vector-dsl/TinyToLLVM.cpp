//===- TinyToLLVM.cpp - Lower Tiny memory ops to LLVM -----------*- C++ -*-===//
//
// This pass lowers Tiny dialect memory operations (load/store) and the ptr
// type to the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "ch1-cpu-vector-dsl/TinyPasses.h"
#include "ch1-cpu-vector-dsl/TinyDialect.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tiny {

// Generate pass definitions.
#define GEN_PASS_DEF_TINYTOLLVM
#include "ch1-cpu-vector-dsl/TinyPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

/// Type converter that converts tiny.ptr to llvm.ptr.
class TinyToLLVMTypeConverter : public TypeConverter {
public:
  TinyToLLVMTypeConverter() {
    // Identity conversion for all types (fallback).
    addConversion([](Type type) { return type; });

    // Convert tiny.ptr to llvm.ptr.
    addConversion([](PtrType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
  }
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Lower tiny.load to llvm.getelementptr + llvm.load.
///
/// tiny.load %ptr, %offset : vector<Nxf16>
/// ->
/// %gep = llvm.getelementptr %ptr[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f16
/// %result = llvm.load %gep : !llvm.ptr -> vector<Nxf16>
///
/// The offset in tiny.load is in f16 elements. The GEP with f16 element type
/// automatically computes the correct byte offset (offset * sizeof(f16)).
struct LoadOpToLLVMLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // The adaptor provides the converted operands (ptr is now !llvm.ptr).
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    // Convert index offset to i64 for GEP.
    Type i64Type = rewriter.getI64Type();
    Value offsetI64 =
        arith::IndexCastOp::create(rewriter, loc, i64Type, offset);

    // Create GEP with f16 element type to compute the address.
    // GEP %ptr[%offset] with f16 element type computes ptr + offset * sizeof(f16).
    Type f16Type = rewriter.getF16Type();
    Type llvmPtrType = LLVM::LLVMPointerType::get(getContext());
    Value gep = LLVM::GEPOp::create(rewriter, loc, llvmPtrType, f16Type, ptr,
                                    ValueRange{offsetI64});

    // Load the vector from the computed address.
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), gep);
    return success();
  }
};

/// Lower tiny.store to llvm.getelementptr + llvm.store.
///
/// tiny.store %vec, %ptr, %offset : vector<Nxf16>
/// ->
/// %gep = llvm.getelementptr %ptr[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, f16
/// llvm.store %vec, %gep : vector<Nxf16>, !llvm.ptr
///
/// The offset in tiny.store is in f16 elements. The GEP with f16 element type
/// automatically computes the correct byte offset (offset * sizeof(f16)).
struct StoreOpToLLVMLowering : public OpConversionPattern<StoreOp> {
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // The adaptor provides the converted operands (ptr is now !llvm.ptr).
    Value value = adaptor.getValue();
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    // Convert index offset to i64 for GEP.
    Type i64Type = rewriter.getI64Type();
    Value offsetI64 =
        arith::IndexCastOp::create(rewriter, loc, i64Type, offset);

    // Create GEP with f16 element type to compute the address.
    Type f16Type = rewriter.getF16Type();
    Type llvmPtrType = LLVM::LLVMPointerType::get(getContext());
    Value gep = LLVM::GEPOp::create(rewriter, loc, llvmPtrType, f16Type, ptr,
                                    ValueRange{offsetI64});

    // Store the vector to the computed address.
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, value, gep);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TinyToLLVM pass implementation
//===----------------------------------------------------------------------===//

class TinyToLLVMPass : public impl::TinyToLLVMBase<TinyToLLVMPass> {
public:
  void runOnOperation() override {
    // Set up the type converter.
    TinyToLLVMTypeConverter typeConverter;

    // Set up the conversion target.
    ConversionTarget target(getContext());

    // Mark memory operations as illegal.
    target.addIllegalOp<LoadOp, StoreOp>();

    // Mark LLVM and arith dialects as legal.
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    // Mark func dialect operations as dynamically legal if their types are
    // converted.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    // Set up rewrite patterns.
    RewritePatternSet patterns(&getContext());

    // Add conversion patterns that use the type converter.
    patterns.add<LoadOpToLLVMLowering, StoreOpToLLVMLowering>(typeConverter,
                                                              &getContext());

    // Add function signature conversion patterns.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Apply the conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tiny
