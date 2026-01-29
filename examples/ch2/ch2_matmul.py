"""Example: Tiled vectorized matrix multiplication."""

from tiny.ch2 import Ptr, Index, F16Vector, accumulate, compile_and_print


@compile_and_print
def matmul(a: Ptr, b: Ptr, c: Ptr, M: Index, N: Index, K: Index):
    """Vectorized matrix multiply: C[M,N] = A[M,K] * B[K,N]^T

    B is transposed so both A and B have contiguous K dimension for vectorization.
    """
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
