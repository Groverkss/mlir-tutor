// Example 1: Linalg Generic to Loops
//
// This example shows how linalg.generic operations map to loops.
//
// RUN: tutorial-opt --convert-linalg-to-loops %s

#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>

// Matrix multiplication: C[m,n] += A[m,k] * B[k,n]
// Iteration space: (m, n, k) where m,n are parallel and k is reduction
func.func @matmul(%A: memref<64x128xf32>, %B: memref<128x64xf32>,
                  %C: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map_A, #map_B, #map_C],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : memref<64x128xf32>, memref<128x64xf32>)
    outs(%C : memref<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
  }
  return
}

// After --convert-linalg-to-loops, this becomes:
//
// scf.for %m = 0 to 64 {
//   scf.for %n = 0 to 64 {
//     scf.for %k = 0 to 128 {
//       %a = memref.load %A[%m, %k]
//       %b = memref.load %B[%k, %n]
//       %c = memref.load %C[%m, %n]
//       %prod = arith.mulf %a, %b
//       %sum = arith.addf %c, %prod
//       memref.store %sum, %C[%m, %n]
//     }
//   }
// }

// -----------------------------------------------------------------------------
// Element-wise addition: C[i,j] = A[i,j] + B[i,j]
// -----------------------------------------------------------------------------

#map_elem = affine_map<(i, j) -> (i, j)>

func.func @elemwise_add(%A: memref<64x64xf32>, %B: memref<64x64xf32>,
                        %C: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map_elem, #map_elem, #map_elem],
    iterator_types = ["parallel", "parallel"]
  } ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
    outs(%C : memref<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  }
  return
}

// -----------------------------------------------------------------------------
// Row-wise sum (reduction): out[i] = sum_j(A[i,j])
// -----------------------------------------------------------------------------

#map_in = affine_map<(i, j) -> (i, j)>
#map_out = affine_map<(i, j) -> (i)>

func.func @row_sum(%A: memref<64x128xf32>, %out: memref<64xf32>) {
  linalg.generic {
    indexing_maps = [#map_in, #map_out],
    iterator_types = ["parallel", "reduction"]
  } ins(%A : memref<64x128xf32>) outs(%out : memref<64xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %sum = arith.addf %a, %acc : f32
      linalg.yield %sum : f32
  }
  return
}
