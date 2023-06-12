
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
    #eta = 0.2
    # third step: find principal directions through PCA
    # and align the data along the principal directions (not done for the moment)
    #idx, _ = pca_alg(data, 2)

    #dpres = idx[:ndim]  # dimensions to be preserved
    #dscal = idx[ndim:] # dimensions to be discarded

    #pca = PCA(n_components=3)
    #pca.fit(data)
    #x_pca = pca.transform(data)

    _, U = pca_alg(data, 3) # keep all dimensions to see just the rotation
    x_pca =  copy.deepcopy(data) # np.dot(data, U)# copy.deepcopy(data)# np.dot(X, U)

    dpres = [0,1]
    dscal = [2]

    prev_data = copy.deepcopy(x_pca)
    # fourth step: iteratively transform data until stop criterion is met
    # stop when all sum of changes for all the points is less than a threshold
    # or if the maximum number of iterations is reached

    for i in tqdm(range(max_iter)):

        # 4a: scale the data along the discarded dimensions
        for j in range(x_pca.shape[0]):
            x_pca[j, dscal] *= sigma

        # The values in Dpres are scaled up to keep 
        # the average neighbor distance equal to avg_dist
        while get_avg_dist(x_pca, neighbors) < avg_dist0:
            for j in range(x_pca.shape[0]):
                x_pca[j, dpres] /= sigma

        
        # create queue of points 
        q = queue.Queue()

        # add a random point to the queue
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
                s, x_pca = adjust_points(x_pca, curr_idx, eta, avg_dist0, dpres, neighbors, dist0, colinear, angles0, adj_data)
                
                step += s
                # add current point to adjusted points
                adj_data.append(curr_idx)

                # add neighbors to queue
                for n in neighbors[curr_idx, :]:
                    q.put(int(n))


        if step >= x_pca.shape[0]:
            eta /= 0.9
        else:
            eta *= 0.9

        # stop criterion
        # if data has not changed much, stop
        change = np.sum(np.abs(x_pca - prev_data))
        prev_data = copy.deepcopy(x_pca)

        if change < th:
            print("Converged after {} iterations".format(i))
            break
        
        if i % 30 == 0:
            print("Iteration: {}, change: {}".format(i, change))

    # final step: project points by dropping the discarded dimensions
    return x_pca#[:, dpres]



def compute_error(data, curr_idx, avg_dist, neighbors, dist0, colinear, angles0, adj_data):
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

        #new_angle = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) if (np.linalg.norm(v1) * np.linalg.norm(v2)) != 0 else 0

        try:
            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                new_angle = 0
            else:
                new_angle = np.arccos(np.clip(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1,1))
        except:
            print('v1: ', v1)
            print('v2: ', v2)
            print('np.linalg.norm(v1): ', np.linalg.norm(v1))
            print('np.linalg.norm(v2): ', np.linalg.norm(v2))
            print('np.dot(v1,v2): ', np.dot(v1,v2))

        error_dist = (( dist0[curr_idx, i] - new_dist)/(2*avg_dist))**2 
        error_angle  = ((angles0[curr_idx, i] - new_angle)/np.pi)**2
        
        error += w *( error_dist + error_angle )

    return error



def adjust_points(data, curr_idx, eta, avg_dist, dpres, neighbors, dist0, colinear, angles0, adj_data):
    s = 0
    improved = True

    eta = 0.3*eta
    #print('prev_point: ', data[curr_idx, :])
    while improved: # until we are in a local minimum
        s += 1
        improved = False

        error = compute_error(data, curr_idx, avg_dist, neighbors, dist0, colinear, angles0, adj_data)

        #print('error: ', error)
        for d in dpres:
            data[curr_idx, d] += eta
            #print('up point: ', data[curr_idx, :])
            
            new_error = compute_error(data, curr_idx, avg_dist, neighbors, dist0, colinear, angles0, adj_data)
            # empirically moving along one of the direction to be preserved
            # choosing the versus by evaluating how the error changes moving both up and down
            if new_error > error:
                #print('new_error ', new_error)
                data[curr_idx, d] -= 2*eta
                #print('down point: ', data[curr_idx, :])
                
                new_error = compute_error(data, curr_idx, avg_dist, neighbors, dist0, colinear, angles0, adj_data)
                if  new_error > error:
                    #print('new_error ', new_error)
                    data[curr_idx, d] += eta
                    #print('up point: ', data[curr_idx, :])
                else:
                    improved = True
            else:
                improved = True
    #print('post_point: ', data[curr_idx, :])

    return s, data
