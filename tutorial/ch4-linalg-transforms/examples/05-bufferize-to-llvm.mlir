// Example 5: Bufferization and LLVM Lowering
//
// This example extends example 4 with bufferization and lowering passes.
// The full pipeline: tile -> fuse -> vectorize -> bufferize -> LLVM
//
// Bufferization converts tensors to buffers (memrefs).
//
// RUN: tutorial-opt --transform-interpreter %s
// RUN: tutorial-opt --transform-interpreter \
// RUN:     --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     --convert-linalg-to-loops \
// RUN:     --convert-scf-to-cf \
// RUN:     --convert-vector-to-llvm \
// RUN:     --convert-func-to-llvm \
// RUN:     --reconcile-unrealized-casts %s

#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>
#map_elem = affine_map<(i, j) -> (i, j)>

func.func @matmul_bias(%A: tensor<64x128xf32>, %B: tensor<128x64xf32>,
                       %bias: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>)
      -> tensor<64x64xf32>

  %matmul = linalg.generic {
    indexing_maps = [#map_A, #map_B, #map_C],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
    outs(%C_init : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
  } -> tensor<64x64xf32>

  %result = linalg.generic {
    indexing_maps = [#map_elem, #map_elem, #map_elem],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%empty : tensor<64x64xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      linalg.yield %sum : f32
  } -> tensor<64x64xf32>

  return %result : tensor<64x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // ========== Tiling and Fusion (same as example 3/4) ==========

    %add = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %tiled_add, %forall = transform.structured.tile_using_forall %add
        tile_sizes [16, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused_matmul, %new_forall = transform.structured.fuse_into_containing_op
        %matmul into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %fused_fill, %final_forall = transform.structured.fuse_into_containing_op
        %fill into %new_forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %inner_matmul = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<parallel>,
                                     #linalg.iterator_type<reduction>]} in %final_forall
        : (!transform.any_op) -> !transform.any_op
    %tiled_k, %k_loop = transform.structured.tile_using_for %inner_matmul
        tile_sizes [0, 0, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ========== Vectorization (same as example 4) ==========

    %func = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op

    // ========== NEW: Bufferization ==========
    // Note: Bufferization is typically done via passes rather than transform ops.
    // After running --transform-interpreter, run these passes:
    //   --one-shot-bufferize="bufferize-function-boundaries"
    //   --convert-linalg-to-loops
    //   --convert-scf-to-cf
    //   --convert-vector-to-llvm
    //   --convert-func-to-llvm
    //   --reconcile-unrealized-casts

    transform.yield
  }
}

// The full lowering pipeline converts:
//
// 1. tensor operations -> memref operations (bufferization)
//    tensor.empty() -> memref.alloc()
//    tensor.extract_slice -> memref.subview
//    tensor.insert_slice -> in-place update
//
// 2. linalg operations -> loops (convert-linalg-to-loops)
//    linalg.generic -> scf.for loops with scalar operations
//
// 3. structured control flow -> unstructured (convert-scf-to-cf)
//    scf.for -> cf.br, cf.cond_br
//
// 4. vector operations -> LLVM intrinsics (convert-vector-to-llvm)
//    vector.transfer_read -> llvm.load with appropriate masking
//    vector.contract -> llvm.fma or similar
//
// 5. func operations -> LLVM (convert-func-to-llvm)
//    func.func -> llvm.func
//    func.return -> llvm.return
