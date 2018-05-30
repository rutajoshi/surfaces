import numpy as np
# Point cloud library 
import pcl
import pcl.pcl_visualization

# Opencv 
import cv2

# Autolab
from autolab_core import PointCloud
from perception import DepthImage
from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d
from clustering import *

# Load the image into a numpy array
image = np.load('/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/depth_ims_numpy/image_000001.npy')
# Create a DepthImage using the array
di = DepthImage(image)
# Visualize the depth image
vis2d.figure()
vis2d.imshow(di)
# Make a dataset of the image
image = make_dataset(image)
image = image.astype(np.float32)

# Viewer
viewer = pcl.pcl_visualization.CloudViewing()

# Make a PCL type point cloud from the image
p = pcl.PointCloud()
p.from_array(image)
viewer.ShowMonochromeCloud(p)

# Make a segmenter and segment the point cloud.
seg = p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(20)
indices, model = seg.segment()


