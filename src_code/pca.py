import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pca_alg(X, n_components):
        
    # calculate the covariance matrix
    # np.cov takes a matrix whose rows are the variables and columns are the observations
    # so we need to transpose the matrix
    cov_matrix = np.cov(X.astype(float).T)

    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # keep the first n_components eigenvectors,
    # which will made our matrix U
    U = eigenvectors[:,:n_components]

    return idx, U

