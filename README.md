# Pattern Recognition Assignments 3: Unsupervised Learning

## K-means

The k-means problem consists of finding groups of points such that the intra-group variance is minimized, that is, minimizing the sum of the squared distances of each point to the center closest to it. 

**The exact algorithm is as follows:**

 1) Choose a center from among the data points using a uniform random variable about dataset.

 2) For each point x, calculate D (x), which is the distance between x and the nearest center that has already been selected.

 3) Choose a new random point (with uniform random variable) as the new center, using a weighted probability distribution where a point x is chosen with probability proportional to D (x) 2.

 4) Repeat steps 2 and 3 until k centers are selected, n iterations.

 5) Now that the initial centers have been chosen, continue using standard k-means clustering.
 
