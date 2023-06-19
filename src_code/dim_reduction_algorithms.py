
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
import numpy as np



#################
## ISOMAP
#################


def MDS(gram_matrix, K):
    # double check that the matrix has the right shape
    N = gram_matrix.shape[0]
    assert gram_matrix.shape == (N, N), 'dist should be a square matrix, but it\'s {}x{}'.format(gram_matrix.shape)

    # Compute the PC scores from Gram matrix
    w, v = np.linalg.eig(gram_matrix)
    # sort the eigenvalues and eigenvectors
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    
    proj = np.diag(np.sqrt(w[:K])) @ v.T[:K]
    
    return proj

def knn_graph(data, k, mode='connectivity'):
    n = data.shape[0]
    W = lil_matrix((n, n))

    for i in range(n):
        # minkowski distance with p=2 is euclidean distance
        distances = np.sqrt(np.sum((data - data[i, :]) ** 2, axis=1))
        indices = np.argsort(distances)
        # connect k nearest neighbors
        for j in indices[:k + 1]:
            if mode == 'connectivity':
                W[i, j] = 1
            elif mode == 'distance':   # sklearn's Isomap uses this mode!!!
                W[i, j] = distances[j]
    return W



def centering_matrix(N):
    '''
    return the centering matrix of size N
    '''
    return np.eye(N) - (1/N) * np.ones((N, N))


def MDS(gram_matrix, K):
    '''
    Compute the K-dimensional embeddings from the Gram matrix
    :param gram_matrix: the Gram matrix
    :param K: the number of dimensions to reduce to
    '''
    # double check that the matrix has the right shape
    N = gram_matrix.shape[0]
    assert gram_matrix.shape == (N, N), 'dist should be a square matrix, but it\'s {}x{}'.format(gram_matrix.shape)

    
    w, v = np.linalg.eig(gram_matrix)
    # sort the eigenvalues and eigenvectors
    idx = np.argsort(w)[::-1]

    w = w[idx]
    v = v[:, idx]

        
    proj = np.dot(np.diag(np.sqrt(w[:K])), v[:,:K].T)  

     
    return proj, w, v



def isomap_alg(data, n_neighbors):
    # step 1 and 2
    # using euclidean distance find the k nearest neighbors and 
    # construct the neighborhood graph
    graph = knn_graph(data, n_neighbors, mode='distance')
    
    # if there is not just one connected component 
    # connect the components with a very large distance
    n_connected_components, labels = connected_components(graph, directed=False)

    if n_connected_components > 1:
        # For each pair of unconnected components, compute all pairwise distances
        # from one component to the other, and add a connection on the closest pair
        # of samples.
        for i in range(n_connected_components):
            for j in range(i + 1, n_connected_components):
                # Find all points in component i, and all points in component j
                # Compute all pairwise distances
                # Add an edge between the closest two points
                dist =  np.zeros((data.shape[0], data.shape[0])) + np.inf 
                for x in range(data.shape[0]):
                    dist[x,:] = [np.sqrt(np.sum((data[x,:] - data[y, :]) ** 2)) if labels[x]==i and labels[y]==j else np.inf  for y in range(data.shape[0])]

                ii, jj = np.where(dist == np.min(dist) )[0][0], np.where(dist == np.min(dist))[1][0]

                graph[ii, jj] = dist[ii, jj]
                graph[jj, ii] = dist[ii, jj]


    # step 3: compute shortest path distances
    dist_matrix = floyd_warshall(csgraph=graph, directed=False)

    
    N = dist_matrix.shape[0]
    # get gram matrix from distance matrix
    dist_matrix = dist_matrix**2
    gram_from_dist = -(1/2) * centering_matrix(N) @ dist_matrix @ centering_matrix(N)

    # step 4: use MDS to compute embeddings
    n_components = data.shape[1]

    proj, eigenval, eigenvec = MDS(gram_from_dist, n_components)

    return proj.real, eigenval.real, eigenvec.real



#################
## Diffusion Map
#################

def diffusion_map(data, n_components, time=2, mode='distance'):
    #W = knn_graph(X, k, mode).toarray()
    W = pairwise_distances(data, metric='euclidean')
    
    sigma = 2.5
    # affinity matrix
    A = np.exp(-W**2 / (2*sigma**2)) 
    P = A / np.sum(A, axis=1)  

    eigvals, eigvecs = np.linalg.eig(P)

    # sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)[::-1][1:]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvecs = eigvecs.real
    eigvals = eigvals.real


    diffusion_coordinates = np.zeros((data.shape[0], n_components))

    for i in range(data.shape[0]):
        for j in range(n_components):
            diffusion_coordinates[i,j] = (eigvals[j])**time * eigvecs[i,j] 

    return diffusion_coordinates


#############
# PCA
#############

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