import numpy as np

# Given matrices A and B, compute A + B
def problem1 (A, B):
    return A + B

# Given matrices A B and C, compute AB-C
def problem2 (A, B, C):
    return np.dot(A,B) - C

# Given matrices A B and C, compute A *(element-wise) B + transpose of C
def problem3 (A, B, C):
    return A * B + np.transpose(C)

# Given column vectors x and y, compute the innner produce of x and y
def problem4 (x, y):
    return np.dot(np.transpose(x),y)

# Given matrix A, return a matrix with the same dimension as A but with all zeros
def problem5 (A):
    return np.zeros(A.shape)

# Given a matrix A, return a vector with the same number of rows as A but that contains all ones
def problem6 (A):
    return np.ones(A.shape[0])

# Given square matrix A and scaler a, compute A + aI, where I is the identity matrix with the same dimensions as A
def problem7 (A, alpha):
    return A + np.dot(alpha,np.eye(A.shape[0],A.shape[1]))

# Given matrix A and integers i,j, return the jth columns of the ith row of A
def problem8 (A, i, j):
    return A[i,j]

# Given matrix A and integer i, return the sum of all of the entries in the ith row
def problem9 (A, i):
    return np.sum(A[i,:])

# Given matrix A and scalers c and d, compute the arithmetic mean over all entries of A that are between c and d (inclusive)
def problem10 (A, c, d):
    a = A[np.nonzero(A>=c)]
    return np.mean(a[a<=d])

# Given a (n x n) matrix A and integer k, return a (n x k) matrix containing the right-eigenvectors of A corresponding to the k largest eigenvalues of A
def problem11 (A, k):
    # if k is greater than the possible amount of eigenvectors, the max amount of eigenvectors is returned
    return np.linalg.eig(A)[1][:,0:k]

# Given square matrix A and column vector x compute A(inverse)x
def problem12 (A, x):
    return np.linalg.solve(A, x)

# Given square matrix A and row vector x, compute xA(inverse)
def problem13 (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A),np.transpose(x)))
