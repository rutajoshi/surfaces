import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import collections
import operator

def make_dataset(image):
    """
    We assume the image is a 2d matrix, with the depths at pixels (x,y).
    Returns an array with 3 columns (x, y, depth) and as many rows as pixels.
    """
    dataset = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            row = [x, y, image[x][y]]
            dataset.append(row)
    return np.array(dataset)

def cluster(image, k=4):
    """ Use sklearn to cluster the depth values and pixels together.
    If using DepthImage, don't make_dataset. Use the image._data attribute.
    """
    # Make dataset
    X = make_dataset(image)
    # Cluster data
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # Plot things
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()

    # Retrieve centroids
    return y_kmeans

def biggest_cluster(y_kmeans):
    """ Returns the centroid index of the largest centroid.
    This corresponds to the largest flat space in a bin of objects.
    """
    counter = collections.Counter(y_kmeans)
    best_cluster = counter.most_common(1)[0][0]
    return best_cluster

def retrieve_clusters(image, y_kmeans):
    """ Given the original image, group the pixels by cluster and return a dict of depths.
    If using DepthImage, don't make_dataset. Instead use image._data attribute.
    """
    X = make_dataset(image)
    groups = dict()
    for i in range(len(y_kmeans)):
        cluster_index = y_kmeans[i]
        if cluster_index in groups:
            groups[cluster_index].append(X[i][2])
        else:
            groups[cluster_index] = [X[i][2]]
    return groups

def average_depths(clusters):
    """ Take the grouped depth values and average each group.
    Return a dict.
    """
    averages = dict()
    for k, v in clusters.items():
        averages[k] = np.mean(np.array(v))
    return averages

def best_placement(image, y_kmeans):
    """ Returns the best cluster for placing an object.
    This corresponds to the lowest avg depth and the largest cluster.
    Normalize the sizes and depths by the max values to weight them equally.
    """
    clusters = retrieve_clusters(image, y_kmeans)
    depths = average_depths(clusters)
    counter = collections.Counter(y_kmeans)
    sizes = dict(counter)

    # Normalize sizes and averages
    max_size = max(sizes.values())
    norm_sizes = dict([(k, v/max_size) for k,v in sizes.items()])
    max_depth = max(depths.values())
    norm_depths = dict([(k, v/max_depth) for k,v in depths.items()])

    # Pick the cluster whose size and depth value combined are largest
    assert norm_sizes.keys() == norm_depths.keys()
    combined = dict()
    for k in norm_depths.keys():
        combined[k] = norm_depths[k] + norm_sizes[k]
    best_cluster = max(combined.items(), key=operator.itemgetter(1))[0]
    return best_cluster
