import numpy as np
from sklearn.metrics import pairwise_distances


def get_local_relationships(data, k=10):
    dist = np.zeros((data.shape[0], k))
    neighbors = np.zeros((data.shape[0], k), dtype=int)

    # matrix of distances
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
            
            # choose the index of the angle which is nearest to pi
            # because the angle between p-n and n-m is pi if the segment p-n is colinear with n-m
            colinear[i,j] = np.argmin(np.abs(angles - np.pi))
            theta[i,j] = angles[colinear[i,j]]

    return neighbors, dist, colinear, theta, avg_dist



def get_avg_dist(data, neighbors):
    """
    Function to compute the average distance between each point and its neighbors
    input:
        data: dataset
        neighbors: matrix of neighbors
    """
    avg_dist = 0
    k = neighbors.shape[1]
    for i in range(data.shape[0]):
        for j in range(k):
            avg_dist += np.linalg.norm(data[i,:] - data[neighbors[i,j],:])/k

    avg_dist /= data.shape[0]

    return avg_dist




def generate_swiss_roll(n):
    """
    Function to generate swiss roll dataset
    input:
        n: number of samples

    output:
        data: swiss roll dataset

    """
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    for i in range(n):
        t = 8*i/n +2

        x[i]= t*np.sin(t)
        z[i]= t*np.cos(t)
        y[i]= np.random.uniform(-1, 1, 1)*6

    data = np.column_stack((x, y, z))

    return data