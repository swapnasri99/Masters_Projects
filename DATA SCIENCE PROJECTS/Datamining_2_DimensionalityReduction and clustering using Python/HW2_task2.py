import numpy as np

# TODO: Implement Kmeans++ algorithm from scratch
# Hint: You can take help from the kmeans++ algorithm implemented in the file "Clustering_Lab_With_Answers.ipynb". 
# The file is uploaded in nextiLearn. The code is a slightly different organization of that done in class. But the basic ideas are same.
# PLEASE DO NOT CHANGE THE NAMES OF FUNCTIONS AND PRESPECIFIED VARIABLES (for automated testing purpose).

def initialize_centroids(X, k):
    """Initializes centroids using the KMeans++ strategy."""
    np.random.seed(42)
    centroids = []
    first_centroid = X[np.random.randint(X.shape[0])] # Fillup the blank space
    centroids.append(first_centroid)
    for _ in range(1,k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        probabilities = distances ** 2 / np.sum(distances ** 2)
        next_centriod_index = np.random.choice(X.shape[0],p=probabilities)
        next_centriod = X[next_centriod_index]
        centroids.append(next_centriod)
    
    '''Write your code here'''
    
    return np.array(centroids)

def assign_clusters(X, centroids):
    """Assigns each point to the nearest centroid."""
    distances = []
    for c in centroids:
        dist = np.linalg.norm(X-c, axis=1) # Calculate distance
        distances.append(dist)
    
    # TODO: Write the code to get the updated labels
    # Hint: convert distances to an np array, and the use argmin() function to get the labels
    distances = np.array(distances)
    labels = np.argmin(distances,axis=0)
    return labels

def update_centroids(X, labels, k):
    """Recomputes centroids as the mean of assigned points."""
    new_centroids = []
    # TODO: Write your code here
    '''Hints: Iterate k times, and for each label (cluster), find the centroid of the cluster using mean() function. 
    Add the new centroid to the list new_centroids'''
    for i in range(k):
        cluster_points = X[labels == i]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100):
    """Performs KMeans clustering."""
    centroids = initialize_centroids(X,k) # TODO: Fill in the missing arguments
    
    # TODO: Write your code here
    for _ in range(max_iters):
        labels = assign_clusters(X,centroids)
        new_centriods = update_centroids(X,labels,k)

        if np.all(centroids == new_centriods):
            break

        centroids =  new_centriods
    
    return labels