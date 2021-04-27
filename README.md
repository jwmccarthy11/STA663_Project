##### Authors: 

Michael Sarkis, Jack McCarthy

##### Github: 

https://github.com/jwmccarthy11/STA663_Project

##### Contributions:

Michael: 
- Main affinity propagation method
- Report write-up
- Cluster plotting method

Jack:
- Optimized message functions
- Package source + build
- Basketball data example

##### Installation:

1. Ensure <a href="https://visualstudio.microsoft.com/visual-cpp-build-tools/">C++ Build Tools 14</a> or greater is installed on your system.


2. Run the following command in your terminal: 

        pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aff-prop
    
    
3. Import via the following line:

        from aff_prop.aff_prop import affinity_propagation
        
    
4. The plotting function may also be imported via:


        from aff_prop.aff_prop import plot_affinity_clusters
