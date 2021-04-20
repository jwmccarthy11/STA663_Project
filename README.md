##### Overview
- Clustering algorithm
- Each data point is node in network
- Messages sent between nodes
- Data points have "affinity" for neighbors as "exemplars" (similar to cluster centroids)
- Exemplars are iteratively chosen through message-passing procedure between data points

##### Input
- Input is similarity $`s(i, k)`$ for points $`i`$ and $`k`$...
    - i.e. $`s(i, k) = -\Vert x_i - x_k \Vert^2`$
    
- ...and "preferences" $`s(k, k)`$
    - Diagonal elements of $`s`$-matrix
    - Larger $`s(k, k) \rightarrow`$ more likely to be chosen as exemplar
    - Under equal prior preference for all points, scale of shared value determines # of clusters
    
##### Algorithm Setup
- Two types of messages are passed
    1. "Responsibility" $`r(i, k)`$ from $`i`$ to $`k`$
        - "...accumulated evidence for how well-suited point $`k`$ is to serve as the exemplar for point $`i`$..."
        - Accounts for other potential exemplars for $`i`$
    2. "Availability" $`a(i, k)`$ from $`k`$ to $`i`$
        - "...accumulated evidence for how appropriate it would be for point $`i`$ to choose point $`k`$ as its exemplar..."
        - Accounts for support from other points that point $`k`$ should be an exemplar
        - Initialize $`a(i, k) = 0`$
        
##### Algorithm Steps
- At each step...
    1. Update responsibility 
    ```math
    r(i, k) \leftarrow s(i, k) - \underset{k' \text{s.t.} k' \neq k}{\max} \left\{ a(i, k') + s(i, k') \right\}
    ```
    2. Update availability for $i \neq k$:
    ```math
    a(i, k) \leftarrow \min \left\{ 0, r(k, k) + \hspace{-15px} \sum\limits_{i' \text{s.t.} i' \not\in \{i, k\}} \hspace{-10px} \max \{0, r(i', k)\} \right\}
    ```
    
    3. Update self-availability:
    ```math
    a(i, k) \leftarrow \hspace{-15px} \sum\limits_{i' \text{s.t.} i' \neq k} \hspace{-10px} \max \{0, r(i', k)\}
    ```
- Algorithm may terminate after...
    - Fixed iterations
    - Changes in messages fall below threshold
    - Local decisions stay constant for a number of iterations

- For point $`i`$, the value of $`k`$ that maximizes $`a(i, k) + r(i, k)`$ either...
    - If $`i=k`$, identifies $`i`$ as an exemplar
    - If $`i \neq k`$, identifies $`k`$ as the exemplar for point $`i`$