from tqdm import tqdm
from pca import *
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import queue    
import numpy as np
import copy

from utils import get_local_relationships, get_avg_dist


class ManifoldSculpting:

    def __init__(self, n_components, k=10, n_iter=100, sigma = 0.9, th = 10**(-3), align=False, verbose=False, starting_point=None):
        """
        Manifold sculpting algorithm
        input:
            n_components: number of components to keep
            k: number of neighbors
            n_iter: number of iterations
            sigma: scaling factor
            th: threshold to stop the algorithm
            align: if True, align the data along the principal directions
            verbose: if True, print the reconstruction error at each iteration

        output:
            self.embedding: the embedding of the dataset
        """
        self.n_components = n_components
        self.k = k
        self.n_iter = n_iter
        self.sigma = sigma
        self.th = th
        self.embedding = None
        
        self.transformed_data = None 
        self.reconstruction_error = None

        self.verbose = verbose
        self.align = align

        self.starting_point = starting_point

    def fit(self, data):
        '''
        Function which implements the manifold sculpting algorithm.
        It takes the data as input and returns the embedding of the data, by performing the following steps:
        1. find the k nearest neighbors for each point and compute local relationships
        2. find the directions to be preserved and those to be discarded through PCA and 
            (optionally) align the data along the principal directions
        3. iteratively transform data and adjust relationships until stop criterion is met
        4. project the data in the new space
        '''

        self.data = data
        # find the k nearest neighbors for each point
        # compute the relationships between each point and its neighbors
        self.neighbors, self.dist0, self.colinear, self.angles0, self.avg_dist0 = get_local_relationships(data, self.k) 

        eta = copy.deepcopy(self.avg_dist0)
    
        # third step: find principal directions through PCA
        # and (optionally) align the data along the principal directions
        idx, U = pca_alg(self.data, self.data.shape[1]) 
        

        if self.align:
            x_pca = np.dot(self.data, U) 
        else:
            x_pca = copy.deepcopy(self.data) 


        self.dpres = idx[:self.n_components]
        self.dscal = idx[self.n_components:]

        if self.verbose:
            print("Dimensions which are going to be preserved: ", self.dpres)
            print("Dimensions which are going to be discarded: ", self.dscal)
    
        prev_data = copy.deepcopy(x_pca)
    
        """
        Iteratively transform data until stop criterion is met,
        stop when all sum of changes for all the points is less 
        than a threshold or if the maximum number of iterations is reached
        """
        
        for i in range(self.n_iter):

            # 4a: scale the data along the discarded dimensions
            for j in range(x_pca.shape[0]):
                x_pca[j, self.dscal] *= self.sigma

            # the values in Dpres are scaled up to keep 
            # the average neighbor distance equal to avg_dist
            while get_avg_dist(x_pca, self.neighbors) < self.avg_dist0:
                for j in range(x_pca.shape[0]):
                    x_pca[j, self.dpres] /= self.sigma


            # create queue of points and add a random point to the queue
            q = queue.Queue()

            if self.starting_point is not None:
                curr_idx = self.starting_point
            else:
                curr_idx = np.random.randint(0, x_pca.shape[0], 1).item()
    
            q.put(curr_idx)

            # keep list of adjusted points
            adj_data = []

            step = 0 

            # while queue is not empty
            while not q.empty():
                # pick point from queue
                curr_idx = q.get()
                
                if curr_idx not in adj_data: # if current point has not been adjusted jet
                    s, x_pca = self.adjust_points(x_pca, curr_idx, eta, adj_data)
                    
                    step += s
                    # add current point to adjusted points
                    adj_data.append(curr_idx)

                    # add neighbors to queue
                    for n in self.neighbors[curr_idx, :]:
                        q.put(int(n))


            # stop criterion
            # if data has not changed much, stop
            change = np.sum(np.abs(x_pca - prev_data))
            prev_data = copy.deepcopy(x_pca)

            if change < self.th and self.verbose:
                print("Converged after {} iterations".format(i))
                break
            
            if i % 10 == 0 and self.verbose:
                print("Iteration: {}, change: {}".format(i, change))

                # if representations of intermediate steps are desired
                # the following lines can be uncommented
                # if self.data.shape[1] <= 3:
                #    self.get_representation(x_pca, i)
                
        self.performed_iter = i

        # final step: project points by dropping the discarded dimensions
        self.embedding = x_pca[:, self.dpres]
        self.transformed_data = x_pca

        # compute reconstruction error
        self.reconstruction_error = np.sum([self.compute_error(x_pca, i, []) for i in range(self.data.shape[0])])
        
        if self.verbose:
            print("Final reconstruction error: {}".format(self.reconstruction_error))



    def adjust_points(self, data, curr_idx, eta, adj_data):
        s = 0
        improved = True
        
        eta = 0.3*eta
        
        while improved: # until we are in a local minimum
            s += 1
            improved = False

            error = self.compute_error(data, curr_idx, adj_data)

            for d in self.dpres:
                data[curr_idx, d] += eta
                
                new_error = self.compute_error(data, curr_idx, adj_data)
                # empirically moving along one of the direction to be preserved
                # choosing the versus by evaluating how the error changes moving both up and down
                if new_error > error:
                    data[curr_idx, d] -= 2*eta
                    
                    new_error = self.compute_error(data, curr_idx, adj_data)
                    if  new_error > error:
                        data[curr_idx, d] += eta
                    else:
                        improved = True
                else:
                    improved = True

        return s, data
    
    def compute_error(self, data, curr_idx, adj_data):
        c = 10
        error = 0
        # heuristic error value is used to evaluate the current relationships
        # among data points relative to the original relationships

        for i,j in enumerate(self.neighbors[curr_idx, :]):
            # weight of the error
            w = 1 
            # change weight if point j has already been adjusted
            if j in adj_data:
                w = c
            
            # distance between current point and point j
            new_dist = np.linalg.norm(data[curr_idx, :] - data[j, :])
            v1 = data[curr_idx, :] - data[j, :] # dist between curr point and point j
            v2 = data[self.colinear[curr_idx,i], :] - data[j, :] # dist between colinear point with current one through point j and point j

            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                new_angle = 0
            else:
                new_angle = np.arccos(np.clip(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1,1))
        
            error_dist = ((self.dist0[curr_idx, i] - new_dist)/(2*self.avg_dist0))**2 
            error_angle  = ((self.angles0[curr_idx, i] - new_angle)/np.pi)**2
            
            error += w *( error_dist + error_angle )

        return error
    
    def get_representation(self, data, i):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2], c=self.data[:,2], cmap='viridis')
        if self.dscal == 0:
            ax.set_xlim(np.min(self.data[:,0]), np.max(self.data[:,0]))
        elif self.dscal == 1:
            ax.set_ylim(np.min(self.data[:,1]), np.max(self.data[:,1]))
        else:
            ax.set_zlim(np.min(self.data[:,2]), np.max(self.data[:,2]))
                
        
        # change the angle of the axes
        ax.view_init(10, -60)
        filename = str(i)+ '_iterations.png'
        plt.savefig(filename)   # save the figure to file
        plt.close(fig)