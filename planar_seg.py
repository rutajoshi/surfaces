# General imports
import numpy as np
import pdb
import cv2
import sys

# Point cloud library imports
import pcl
import pcl.pcl_visualization

# Autolab imports
from autolab_core import PointCloud
from perception import DepthImage, CameraIntrinsics
from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d
from clustering import *

# Load the image into a numpy array
image = np.load('/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/depth_ims_numpy/image_000001.npy')
# Create a DepthImage using the array
ci = CameraIntrinsics.load('/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/camera_intrinsics.intr')
di = DepthImage(image, frame=ci.frame)
pc = ci.deproject(di)

## Visualize the depth image
#vis2d.figure()
#vis2d.imshow(di)
#vis2d.show()


# Make and display a PCL type point cloud from the image
p = pcl.PointCloud(pc.data.T.astype(np.float32))

# Make a segmenter and segment the point cloud.
seg = p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PARALLEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.005)
indices, model = seg.segment()
print(model)


#pdb.set_trace()
vis3d.figure()
pc_plane = pc.data.T[indices]
pc_plane = pc_plane[np.where(pc_plane[::,1] < 0.16)]

maxes = np.max(pc_plane, axis=0)
mins = np.min(pc_plane, axis=0)
print('maxes are :', maxes ,'\nmins are : ', mins)

vis3d.points(pc_plane, color=(1,0,0))
vis3d.show()

