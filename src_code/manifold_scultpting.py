from tqdm import tqdm
from pca import *
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import queue    
import numpy as np
import copy



def get_local_relationships(data, k=10):
    dist = np.zeros((data.shape[0], k))
    neighbors = np.zeros((data.shape[0], k), dtype=int)

    # matrix of distances
    distances = np.zeros((data.shape[0], data.shape[0]))
    
    """ for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distances[i, j] = np.linalg.norm(data[i, :] - data[j, :]) """
    
    distances = pairwise_distances(data,  metric='euclidean')

    # keep track of average distance between each point and its neighbors
    avg_dist = 0

    # keep only the k nearest neighbors
    for i in range(data.shape[0]):
        neighbors[i, :] = np.argsort(distances[i, :])[1:k+1]
        dist[i, :] = np.sort(distances[i, :])[1:k+1]
        avg_dist += np.sum(dist[i, :]) / k

    avg_dist /= data.shape[0]

    # for each pair point neighbor (p,n) measure the angle between 
    # the segment p-n and n-m where m is the most colinear neighbor of n with p
    theta = np.zeros((data.shape[0], k)) #angles with colinear points
    colinear = np.zeros((data.shape[0], k), dtype=int) #indexes of colinear points

    for i in range(data.shape[0]):
        for j in range(k):
            n = int(neighbors[i, j])
            # find the most colinear neighbor of n with p
            # the most colinear neighbor is the one with the smallest angle
            # vector between p and n
            v1 = data[i,:] - data[n,:]

            # angles between p-n and n-m
            angles = np.zeros(data.shape[0])
            for z in range(k):
                m = int(neighbors[n, z])
                if m != i:
                    # vector between n and m
                    v2 = data[m,:] - data[n,:]
                    angles[m] = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            
            # choose the index of the angle which is neared to pi
            # because the angle between p-n and n-m is pi if the segment p-n is colinear with n-m
            colinear[i,j] = np.argmin(np.abs(angles - np.pi))
            theta[i,j] = angles[colinear[i,j]]

    return neighbors, dist, colinear, theta, avg_dist


def get_avg_dist(data, neighbors):

    avg_dist = 0
    k = neighbors.shape[1]
    for i in range(data.shape[0]):
        for j in range(k):
            avg_dist += np.linalg.norm(data[i,:] - data[neighbors[i,j],:])/k

    avg_dist /= data.shape[0]

    return avg_dist


def manifold_sculpting(data, k = 10, ndim=2, max_iter = 1000, th = 10**(-5)):
    """
    input:
        data: the dataset representing the manifold
        k: number of nearest neighbors to consider
        ndim: number of dimensions of the output space
        max_iter: maximum number of iterations
        th: threshold to stop the algorithm
    output:
        data transformed and projected in the 2D space
    """
    sigma = 0.9 # scaling factor

    # first step: find the k nearest neighbors for each point
    # second step: compute the distance between each point and its neighbors
    neighbors, dist0, colinear, angles0, avg_dist0 = get_local_relationships(data, k) # matrix of distances with only the k nearest neighbors

    eta = copy.deepcopy(avg_dist0)

    # third step: find principal directions through PCA
    # and align the data along the principal directions (not done for the moment)
    #idx, _ = pca_alg(data, 2)

    #dpres = idx[:ndim]  # dimensions to be preserved
    #dscal = idx[ndim:] # dimensions to be discarded

    pca = PCA(n_components=3)
    pca.fit(data)
    x_pca = pca.transform(data)

    dpres = [0,1]
    dscal = [2]

    prev_data = copy.deepcopy(x_pca)
    # fourth step: iteratively transform data until stop criterion is met
    # stop when all sum of changes for all the points is less than a threshold
    # or if the maximum number of iterations is reached

    for i in tqdm(range(max_iter)):

        # 4a: scale the data along the discarded dimensions
        for j in range(data.shape[0]):
            x_pca[j, dscal] *= sigma

        # The values in Dpres are scaled up to keep 
        # the average neighbor distance equal to avg_dist
        while get_avg_dist(x_pca, neighbors) < avg_dist0:
            for j in range(data.shape[0]):
                x_pca[j, dpres] /= sigma

        
        # recompute the neighbors (distances)
        _, dist, _, angles, avg_dist = get_local_relationships(data, k)
        eta = avg_dist

        # create queue of points 
        q = queue.Queue()

        # add a random point to the queue
        curr_idx = np.random.randint(0, data.shape[0], 1).item()
        q.put(curr_idx)

        # keep list of adjusted points
        adj_data = []

        step = 0 

        # while queue is not empty
        while not q.empty():
            # pick point from queue
            curr_idx = q.get()
            
            if curr_idx not in adj_data: # if current point has not been adjusted jet
                step += adjust_points(data, curr_idx, eta, dpres, neighbors, dist0, colinear, angles0, adj_data)
                # add current point to adjusted points
                adj_data.append(curr_idx)

                # add neighbors to queue
                for n in neighbors[curr_idx, :]:
                    q.put(int(n))


        if step >= data.shape[0]:
            eta /= 0.9
        else:
            eta *= 0.9

        # stop criterion
        # if data has not changed much, stop
        change = np.sum(np.abs(data - prev_data))
        prev_data = copy.deepcopy(data)

        if change < th:
            print("Converged after {} iterations".format(i))
            break


    # final step: project points by dropping the discarded dimensions
    return data[:, dpres]



def compute_error(data, curr_idx, eta, neighbors, dist0, colinear, angles0, adj_data):
    c = 10
    error = 0
    # heuristic error value is used to evaluate the current relationships
    # among data points relative to the original relationships

    for i,j in enumerate(neighbors[curr_idx, :]):
        # weight of the error
        w = 1 
        # change weight if point j has already been adjusted
        if j in adj_data:
            w = c
        
        # distance between current point and point j
        new_dist = np.linalg.norm(data[curr_idx, :] - data[j, :])
        v1 = data[curr_idx, :] - data[j, :] # dist between curr point and point j
        v2 = data[colinear[curr_idx,i], :] - data[j, :] # dist between colinear point with current one through point j and point j

        new_angle = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        error += w *( ((new_dist - dist0[curr_idx, i])/(2*eta))**2 
                     + ((new_angle - angles0[curr_idx, i])/(np.pi))**2 )


    return error



def adjust_points(data, curr_idx, eta, dpres, neighbors, dist0, colinear, angles0, adj_data):
    s = 0
    improved = True


    while improved: # until we are in a local minimum
        s += 1
        improved = False

        error = compute_error(data, curr_idx, eta, neighbors, dist0, colinear, angles0, adj_data)

        for d in dpres:
            data[curr_idx, d] += eta
            # empirically moving along one of the direction to be preserved
            # choosing the versus by evaluating how the error changes moving both up and down
            if compute_error(data, curr_idx, eta, neighbors, dist0, colinear, angles0, adj_data) > error:
                data[curr_idx, d] -= 2*eta
                if compute_error(data, curr_idx, eta, neighbors, dist0, colinear, angles0, adj_data) > error:
                    data[curr_idx, d] += eta
                else:
                    improved = True
            else:
                improved = True
    
    return s