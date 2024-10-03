import numpy as np
from scipy.linalg import qr

def economic_qr_with_rank_check(A):
    """
    Perform the economic QR factorization of matrix A, check the rank of R,
    and reduce R if necessary while keeping Q the original size.
    
    Args:
    - A (np.ndarray): The input matrix to factorize.
    
    Returns:
    - Q (np.ndarray): The orthogonal matrix.
    - R (np.ndarray): The upper triangular matrix (possibly reduced).
    - rank_R (int): the column rank of A.
    """
    # Perform QR factorization with mode='economic' for thin QR
    Q, R = qr(A, mode='economic')
    
    # Examine the rank of R
    rank_R = np.linalg.matrix_rank(R)
    print(f"Rank of R: {rank_R}")
    
    # Reduce only R if rank deficient
    if rank_R < R.shape[1]:
        R_reduced = np.zeros_like(R)  # Create a zero matrix of the same size as R
        R_reduced[:rank_R, :] = R[:rank_R, :]  # Keep only the top non-zero rows
        return Q, R_reduced, rank_R
    else:
        return Q, R, rank_R, 