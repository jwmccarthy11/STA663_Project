# Affinity Propagation (aff_prop)

This repository contains all code relevant to the development and distribution to the package "aff_prop", which implements the affinity propagation (AP) clustering algorithm.

### Algorithm Explanation

The affinity propogation algorithm begins by taking an (n x n) matrix of pairwise similarities for all of the points in the data set. The diagonal of this matrix (the "similarity" between a point and itself) is instead taken to be the preference that the specific point is an exemplar. This diagonal is considered to be the points preferences. To begin the responsibility and availability matrices are set to n by n matrices containing only 0's. From here the algorithm iteratively performs the following steps: Updating responsibility, updating availability and updating the chosen exemplars.

The responsibility matrix at point (i,k) is updated to be the the similarity of points $i$ and k minus the maxmimum of similarity plus availability that i has with any other point. To get the complete updated responsibility matrix damping is applied, such that the new responsibility matrix is equal to the old matrix times a weight plus the updated values times 1 minus that weight. The availability matrix at point (i,k), when i $\neq$ k is updated to be the the minimum of 0 and the self-responsibility of k plus the sum of all positive responsibilities that k has with other points. The availability matrix at point (k,k) is updated to be the sum of all positive responsibility that k has with other points. Again, to get the complete updated availability matrix damping is applied, such that the new availaibility matrix is equal to the old matrix times a weight plus the updated values times 1 minus that weight. The exemplars are then updated by summing the current responsibility and availability matrices and  for each row i the column k that has the greatest value represents the exemplar for i. If k = i then i iteself is deemed an exemplar.

This process continues until one of three stopping conditions is reached. The algorithm then returns the indices of the points selected as exemplars, the number of exemplars, the exemplar which each point has clustered with and optionally the iterations taken to reach these results.

### Example

The package functions may be imported as follows

        from aff_prop.aff_prop import affinity_propagation, plot_affinity_clusters

Given the data plotted below

![raw_data](https://user-images.githubusercontent.com/70278753/116308741-de72a680-a775-11eb-8be0-630f1238887d.PNG)

The clustering is done by the following function

        k, exems, labels, _, _ = affinity_propagation(s)
        
and the output may be plotted using the included plotting helper function

        plot_affinity_clusters(c,exems,labels)

![afp_data](https://user-images.githubusercontent.com/70278753/116308755-e03c6a00-a775-11eb-8d4d-3d83393e8e7f.PNG)


### Installation

1. Ensure <a href="https://visualstudio.microsoft.com/visual-cpp-build-tools/">C++ Build Tools 14</a> or greater is installed on your system.


2. Run the following command in your terminal: 

        pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aff-prop
    
    
3. Import via the following line:

        from aff_prop.aff_prop import affinity_propagation
        
    
4. The plotting function may also be imported via:


        from aff_prop.aff_prop import plot_affinity_clusters


MIT &copy; Michael Sarkis, Jack McCarthy
