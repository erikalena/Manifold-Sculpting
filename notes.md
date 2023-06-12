Paper: **Iterative Non-linear Dimensionality Reduction by Manifold Sculpting**

Implementation and comparison with other methods for manifold learning.

Manifold algorithm **iteratively reduces** simensionality by **simulating surface tension in local neighborhoods**.


## Introduction

Dimensionality reduction consists of two steps:

1. transform the data so that more information will survive the projection 
2. project the data onto a lower dimensional space

The more relationships between data points tat the transformation step is required to preserve, the less flexibility it will have to position the points in a manner that will cause information to survive the projection step. 

Due to this inverse relationship, dimensionality reduction algorithms must seek a balance that preserves information in the transformation without losing it in the projection. 

**The key to finding the right balance is to identify where the majority of the information lies.**

NLDR (Non-Linear Dimensionality Reduction) algorithms seek this balance by assuming that the relationships between neighboring points contain more informational content than the relationships between distant points.

Embeddings:

[x1, x2, ..., xn] ->  [f1(x1, x2, ..., xn), f2(x1, x2, ..., xn)]

in non linear methods, the functions f1, f2, ..., fn are non linear.


Although non linear transformations have more potential than do linear transformations to lose information in the structure of the data, they also have more potential to position the data to cause more information to survive the projection.

Manifold Sculpting discovers manifolds through a process of progressive refinement. Experiments show that it yields more accurate results than other algorithms in many cases. Additionally, it can be used as a **postprocessing step** to enhance the transformation of other manifold learning algorithms.



## Related Work 

* Isomap
  - it is computationally expensive because it requires to solving for the eigenvectors of a large matrix and it has difficulty with poorly sampled areas of the manifold.

* LLE:
    - it is able to perform a similar computation (as Isomap) using a sparse matrix by using a metric that measures only relationships between vectors in local neighborhoods.
    - however it produces distorted results when the sample density is non-uniform.

* HLLE: preserves the manifold structure better than LLE, but it is computationally expensive.



## The algorithm


It is robust to sampling issues and still produces very accurate results. It iteratively transforms data by **balancing two opposing heuristics**, one that **scales information out** of unwanted dimensions and another that **preserves information** into fewer dimensions with more accuracy than existing manifold learning algorithms.


Steps:

1. Find the $k$ nearest neighbors of each point
2. Compute a set of relationships between neighbors
3. Optionally align axes with principle components
4. While the stopping criterion has not been met...

    a. scale the data in the non preserved dimensions

    b. adjust points to restore the relationships between neighbors
5. Project the data

**Step 1:** find the $k$ nearest neighbors of each point. For each data point $p_i$ find the set $N_i$ of k-nearest neighbors, $n_{ij}$ is the $jth$ neighbor of $p_i$.

**Step 2:** compute the relationship between the found neighbors. For each point $p_i$ compute the set of relationships $R_i$ between its neighbors. The relationship between $p_i$ and $n_{ij}$ is given by Euclidean disance $\delta_{ij}$.

The angle $\theta_{ij}$ between the two lines segments ($p_i$ to $n_{ij}$) and ($n_{ij}$ to $m_{ij}$) is computed as well. $m_{ij}$ is the most colinear neighbor of $n_{ij}$ with respect to $p_i$.

That is to say the neighbor point that forms the angle closest to $\pi$. 

The values of $\delta$ and $\theta$ are the relationships that the algorithm will attempt to preserve.

The global average distance between all the neighbors of all points $\delta_{average}$ is computed as well.

**Step 3**: the data may optionally be preprocessed with the transformation step of Principle Component Analysis (PCA), or another efficient algorithm.

In this way, the algorithm could converge faster. 

To the extent that there is a linear component in the manifold, PCA will move the information in the data into as few dimensions as possible, leaving less work to be done in step 4, which handles the non linear component.

This step is performed by computing the first $|D_{pres}|$ principle components of the data, where $D_{pres}$ is the set of dimensions that will be preserved in the projection and rotating the dimensional axes to align with these principle components.


 **Step 4**: data are transformed. Iterative transformation until stopping criterion has been met (e.g. stop when the sum change of all points during the current iteration falls below a threshold).

- 4a. **scale values**. all the values in $D_{scal}$, which is the set of dimensions that will be eliminated by the projection are scaled by a constant factor $\sigma$, such that $D_{scal}$ over time will converge to 0 and when $D_{scal}$ is dropped by the projection (step 5) there will be very little information content left in these dimensions.

- 4b. **Restore original relationships**. For each $p_i \in P$, the values in $D_{pres}$ are adjusted to recover the relationships distorted by scaling performed at step 4a.

    **Intuitively this step simulates tension on the manifold surface.**

    A heuristic error values is used to evaluate the current relationships (at time $t$) among data points relative to the original relationships:

$$\epsilon_{p_i}^t = \sum_{j=0}^{k} w_{ij} \Big( \Big(\cfrac{\delta_{ij}^t - \delta_{ij}^0}{2\delta_{ave}}\Big)^2 + \Big(\cfrac{\theta_{ij}^t - \theta_{ij}^0}{\pi}\Big)^2 \Big)$$


$\delta_{ij}^t$: current distance to $n_{ij}$,

$\delta_{ij}^0$: original distance to $n_{ij}$, measured at step 2,

$\theta_{ij}^t$: current angle,

$\theta_{ij}^0$: original angle, measured at step 2.

The denominator values were chosen instead as normalizing factors to make the error values comparable. The value of the angle term can range from $0$ to $\pi$, and the value of distance term will have a mean of of $\delta_{ave}$ with some variance in both directions.

**We adjust the values in $D_{pres}$ to minimize the error values.**

The values in $D_{pres}$ are our parameters, they are those which are updated by gradient descent.

The order in which points are asjusted has some impact on the rate of convergence.

Best results were obtained by employing a breadth-first neighborhood graph traversal from a randomly selected point. 

To further speed convergence, heigher weight $w_{ij}$ is given to the component of the error contributed by neighbors that have already been adjusted in the current iteration.

For the experiments, they used $w_{ij} = 1$ if $n_{ij}$ has not yet been adjusted and $w_{ij} = 10$ if $n_{ij}$ has already been adjusted.

**The equation for the true gradient of the error surface** (wrt ours parameters) **defined by this heuristic is complex,** it is in $O(|D|^3)$, where $|D|$ is the number of dimensions in the original data.



Why it is $O(|D|^3)$?

Because the error surface is defined by the sum of the squared errors of each neighbor, and each neighbor is a function of the values in $D_{pres}$, which is a function of the values in $D$, which is a function of the original data.

**Therefore, they used the simple hill-climbing technique of adjusting in each dimension in the direction that yields improvement.** (Instead of taking the derivative  wtr $\Delta_{pres}$, take the derivative with respect to each dimension in $\Delta_{pres}$ and update values of $\Delta_{pres}$ in that direction).

Every point has a unique error surface.


**Step 5**: project the data. The data are projected onto the subspace defined by $D_{pres}$, and the values in $D_{scal}$ are dropped.