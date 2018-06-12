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
"""

""" 1. Given depth image of bin, retrieve largest planar surface """
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
    return indices, model, image, pc

# Visualize a point cloud at certain indices only
def visualize(pc, indices, color=(1,0,0)):
    vis3d.figure()
    pc_plane = pc.data.T[indices]
    pc_plane = pc_plane[np.where(pc_plane[::,1] < 0.16)]
    vis3d.points(pc_plane, color=(1,0,0))
    vis3d.show()

""" 2. Partition the point cloud of the bin into a 2d grid """
def partition_surface(pc, indices, model, xsplits, ysplits):
    return None

""" 3. Given an object mesh, find stable poses """
def find_stable_poses(mesh_file):
    # Load the mesh and compute stable poses
    mesh = trimesh.load(mesh_file)
    stable_poses, probs = mesh.compute_stable_poses()
    assert len(stable_poses) == len(probs) 
    # Find the most probable stable pose
    i, value = max(enumerate(probs), key=operator.itemgetter(1))
    best_pose = stable_poses[i]
    # Visualize the mesh in the most probable stable state
    # rotation = np.asarray(best_pose[:3, :3])
    # translation = np.asarray(best_pose[:3, 3])
    # rt = RigidTransform(rotation, translation, from_frame='obj', to_frame='world')
    # vis3d.mesh(mesh, rt)
    # vis3d.show()
    return mesh, rt, best_pose

""" 4. Given the object in the stable pose, find the footprint on a given cell given the cell plane """
def find_footprint(mesh, rt, best_pose, plane_normal):
    ch = mesh.convex_hull
    ch = ch.apply_transform(best_pose)
    faces_to_keep, fn_to_keep = [], []
    for face, face_normal in zip(ch.faces, ch.face_normals):
        dot_prod = np.dot(face_normal, plane_normal)
        if dot_prod < 1:
            faces_to_keep.append(face)
            fn_to_keep.append(face_normal)
    shadow = ch.copy()
    shadow.faces = faces_to_keep
    shadow.face_normals = fn_to_keep
    shadow.fix_normals()
    return shadow

""" 5. Score each cell given the footprint and other metrics """
def score_cells(pc, indices, model, shadow):
    length, width, height = shadow.extents()
    split_size = max(length, width)
    # go through all the cells, assuming that the mesh is on the same scale as the bin
    for i in range(

""" 6. Return the cell with the highest score """
