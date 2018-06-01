# General imports
import numpy as np
import pdb
import cv2

# Point cloud library 
import pcl
import pcl.pcl_visualization

# Autolab
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
vis2d.figure()
vis2d.imshow(di)
vis2d.show()
## Visualize the point cloud
#vis3d.figure()
#vis3d.points(pc)
#vis3d.show()


# Make and display a PCL type point cloud from the image
p = pcl.PointCloud(pc.data.T.astype(np.float32))

# Make a segmenter and segment the point cloud.
seg = p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PARALLEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.005)
indices, model = seg.segment()

#pdb.set_trace()
vis3d.figure()
pc_plane = pc.data.T[indices]
#others = np.arange(len(pc.data.T))
#other_indices = np.where(others not in indices)
#other_indices = np.array([i for i in range(len(pc.data.T)) if i not in indices])
#pc_other = pc.data.T[other_indices]
vis3d.points(pc_plane, color=(1,0,0))
#vis3d.points(pc_other, color=(0,1,0))
vis3d.show()

