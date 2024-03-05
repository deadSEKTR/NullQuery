import numpy as np

def null_space(matrix, tol=1e-15):
    u, s, vh = np.linalg.svd(matrix)
    null_mask = (s <= tol)
    null_basis = vh[null_mask].T
    null_dimension = np.sum(null_mask)
    return null_dimension, null_basis

def left_null_space(matrix, tol=1e-15):
    u, s, vh = np.linalg.svd(matrix)
    left_null_mask = (s <= tol)
    left_null_basis = u[:, left_null_mask]
    left_null_dimension = np.sum(left_null_mask)
    return left_null_dimension, left_null_basis

matrix = np.array([[3, 2, 0, 0],
                   [0, 5, 0, 0],
                   [1, 0, 3, 0],
                   [0, 0, 0, 0]])

null_dim, null_basis = null_space(matrix)
left_null_dim, left_null_basis = left_null_space(matrix)

print("Null Space:")
print("Dimension:", null_dim)
print("Basis:")
print(null_basis)

print("\nLeft Null Space:")
print("Dimension:", left_null_dim)
print("Basis:")
print(left_null_basis)
