import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import cg as cupy_cg
from cupyx.profiler import benchmark

import jlinops



# Grid size
N_x = 1000
N_y = 1000
img_shape = (N_x, N_y)

# Build operators both as matrix and as black box
Lmat = jlinops.build_neumann2d_sparse_matrix(img_shape)
L = jlinops.Neumann2D(img_shape)


def Lpinv_factory(w, *args, **kwargs):
    return jlinops.CGWeightedNeumann2DPinvOperator(img_shape, weights=w, tol=1e-8, *args, **kwargs)


# Random weight vector
w = cp.random.uniform(size=L.shape[0], low=1, high=10)

# Build Pinv operator
Lpinv = Lpinv_factory(w)
Lpinv = Lpinv.to_gpu()

# Apply to a random vector
# rhs = cp.random.normal(size=L.shape[0])
# _ = Lpinv.matvec(rhs)

def my_func():
    rhs = cp.random.normal(size=L.shape[0])
    _ = Lpinv.matvec(rhs)
    _ = cp.asnumpy(_)

#print(benchmark(my_func, (), n_repeat=20))  

# Use benchmark to time the function
results = benchmark(my_func, n_repeat=25)


# Print the timing result in terms of seconds
print(f"GPU function execution time (average): {results.gpu_times.mean()} seconds")


# Use benchmark to time the function
#results = benchmark(gpu_function, n_repeat=100)



# A = cp.random.normal(size=(100,50))
# A = A.T @ A
# b = cp.random.normal(size=50)
# #_, _ = cupy_cg(A, b, M=None, x0=None)

# import cupy as cp
# from cupyx.scipy.sparse import csr_matrix
# from cupyx.scipy.sparse.linalg import cg

# # Create a symmetric positive-definite sparse matrix A (CSR format)
# # For example, a simple 3x3 matrix
# A_dense = cp.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=cp.float32)
# A_dense = A_dense.T @ A_dense + cp.eye(3)
# A_sparse = csr_matrix(A_dense)  # Convert to sparse matrix (CSR format)

# # Create a vector b
# b = cp.array([1, 2, 3], dtype=cp.float32)

# # Solve the system Ax = b using the Conjugate Gradient method
# x, info = cg(A_sparse, b)

# # Display the solution
# print(f"Solution x: {x}")
# print(f"Convergence info: {info}")  # 0 means successful convergence
