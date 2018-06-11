# General imports
import numpy as np
import pdb
import cv2
import trimesh
import operator

# Point cloud library imports
import pcl
import pcl.pcl_visualization

# Autolab imports
from autolab_core import PointCloud, RigidTransform
from perception import DepthImage, CameraIntrinsics
from visualization import Visualizer2D as vis2d, Visualizer3D as vis3d
from clustering import *

"""
This library uses an object mesh file and a depth image of a bin for object placement to infer the best placement location for the given object in the
bin from the depth image. 

@author: Ruta Joshi
@date: June 11, 2018
"""

# 1. Given depth image of bin, retrieve largest planar surface
def largest_planar_surface(filename, cam_int_file):
    # Load the image as a numpy array and the camera intrinsics
    image = np.load(filename)
    ci = CameraIntrinsics.load(cam_int_file)
    # Create and deproject a depth image of the data using the camera intrinsics
    di = DepthImage(image, frame=ci.frame)
    pc = ci.deproject(di)
    # Make a PCL type point cloud from the image
    p = pcl.PointCloud(pc.data.T.astype(np.float32))
    # Make a segmenter and segment the point cloud.
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PARALLEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.005)
    indices, model = seg.segment()
    return indices, image, pc

# Visualize a point cloud at certain indices only
def visualize(pc, indices, color=(1,0,0)):
    vis3d.figure()
    pc_plane = pc.data.T[indices]
    vis3d.points(pc_plane, color=(1,0,0))
    vis3d.show()

# 2. Partition the point cloud of the bin into a 2d grid
def partition_surface(pc, indices, xsplits, ysplits):
    return None

# 3. Given an object mesh, find stable poses
def find_stable_poses(mesh_file):
    # Load the mesh and compute stable poses
    bc = trimesh.load(mesh_file)
    stable_poses, probs = bc.compute_stable_poses()
    assert len(stable_poses) == len(probs) 
    # Find the most probable stable pose
    i, value = max(enumerate(probs), key=operator.itemgetter(1))
    best_pose = stable_poses[i]
    # Visualize the mesh in the most probable stable state
    # rotation = np.asarray(best_pose[:3, :3])
    # translation = np.asarray(best_pose[:3, 3])
    # rt = RigidTransform(rotation, translation, from_frame='obj', to_frame='world')
    # vis3d.mesh(bc, rt)
    # vis3d.show()

# 4. Given the object in the stable pose, find the footprint on a given cell

# 5. Score each cell given the footprint and other metrics

# 6. Return the cell with the highest score
