## Task 5
# Run the clustering algorithms
# TODO: Run both the clustering algortihms for both spherical and non-sperical data (so in total 4 function calls)
kmeans_labels_spherical = kmeans(spherical_data, k=3)
kmeans_labels_nonspherical = kmeans(nonspherical_data, k=2)
agglomerative_labels_spherical = agglomerative_clustering(spherical_data, k=3)
agglomerative_labels_nonspherical = agglomerative_clustering(nonspherical_data, k=2)


# Compute and print the evaluation metrics
# TODO: Compute both the silhoutte scores and purity scores - each for both kmeans++ algorithm and aglomerative algorithm, for both datasets.
# So in total, 8 function calls. 
# Print all the 8 values.

print(silhouette_score(spherical_data, kmeans_labels_spherical))
print(silhouette_score(nonspherical_data, kmeans_labels_nonspherical))
print(silhouette_score(spherical_data,agglomerative_labels_spherical))
print(silhouette_score(nonspherical_data, agglomerative_labels_nonspherical))
print(purity_score(spherical_labels,kmeans_labels_spherical))
print(purity_score(nonspherical_labels,kmeans_labels_nonspherical))
print(purity_score(spherical_labels,agglomerative_labels_spherical))
print(purity_score(nonspherical_labels,agglomerative_labels_nonspherical))


'''Food for thought and further scope of learning (optional, but recommended)
- Which algorithms work (and not work) for the different datasets? Why?
- Try implementing DBSCAN clustering algorithm and compare the performances
- Try the same exercises by taking a higher dimensional data, performing dimensionality reduction, running clustering algorithms, and comparing performances'''