from clustering import *
from perception import DepthImage
from visualization import Visualizer2D as vis2d

arr = np.load('/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/depth_ims_numpy/image_000000.npy')
di = DepthImage(arr)
vis2d.imshow(di)
image = di._data
# print(image)
X = make_dataset(image)
y_kmeans = cluster(image, k=10)
#print("Y-kmeans: " + str(y_kmeans))
biggest = biggest_cluster(y_kmeans)
print("Biggest cluster: " + str(biggest))
clusters = retrieve_clusters(image, y_kmeans)
#print(clusters)
#print("Retrieved clusters.")
avg_depths = average_depths(clusters)
print(avg_depths)
print("Retrieved avg depths.")
best = best_placement(image, y_kmeans)
print("Best cluster: " + str(best))
