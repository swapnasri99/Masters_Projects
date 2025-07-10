import numpy as np
from scipy.spatial.distance import pdist, squareform

# TODO: Fill up the missing codes, blank spaces
from scipy.spatial.distance import pdist, squareform

def compute_distance_matrix(X):
    """Computes the pairwise distance matrix."""
    pairwise_distances = pdist(X) # TODO, Hint: Compute all pairwise distances using pdist() function
    distance_matrix = squareform(pairwise_distances)   # TODO, Hint: Convert to square matrix format
    return distance_matrix

def find_closest_clusters(distances):
    """Finds the indices of the two closest clusters."""
    min_dist_index =  np.argmin(distances)  # TODO, Hint: Get index of the smallest value using argmin() function
    i, j = i, j = np.unravel_index(min_dist_index,distances.shape)   # TODO, Hint: Convert to row, col indices using np.unravel() function with appropriate arguments
    return i, j

def update_distances(distances, clusters, i, j):
    """Updates the distance matrix after merging clusters i and j."""
    for idx in clusters:
        if idx != i:  # Skip the merged cluster
            cluster_distances = []  # Create an empty list to store distances
            # TODO: Loop through each point in cluster i
                # TODO: Append the corresponding distance
            # TODO: Update "distances" using Minimum linkage
            for x in clusters[i]:
                for y in clusters[idx]:
                    cluster_distances.append(distances[x,y])
            distances[i][idx] = distances[idx][i] = min(cluster_distances)
    distances[:, j] = distances[j, :] = np.inf  # Set merged cluster to infinity to ignore it

def agglomerative_clustering(X, k):
    """Performs agglomerative hierarchical clustering using minimum distance linkage."""
    num_points = len(X)
    
    # Initialize each point as its own cluster
    clusters = {}  # Create an empty dictionary
    for i in range(num_points):  # Loop through each point index
        clusters[i] = [i]  # Assign each point to its own cluster
    
    # TODO: Compute distances matrix using the function compute_distance_matrix() defined earlier
    # TODO: Set diagonal to infinity to ignore self-distances
    distances = compute_distance_matrix(X)
    np.fill_diagonal(distances,np.inf)
    
    while len(clusters) > k:
        i, j = find_closest_clusters(distances) # Get the closest clusters using the function find_closest_clusters(distances)
        
        # Merge cluster j into cluster i
        clusters[i].extend(clusters[j])
        del clusters[j]
        
        # TODO: Update the distance matrix using the function update_distances() defined earlier
        update_distances(distances,clusters,i,j)
    
    # Assign labels to points based on final clusters
    labels = np.zeros(num_points)
    # TODO: Complete the code here to get the labels. Use hints below:
    ''' enumerate over clusters values, and for each point, assign the corresponding labor as equal to the cluster id'''
    for c_id, points in enumerate(clusters.values()):
        for point in points:
            labels[point]= c_id
    
    return labels