# General imports
from __future__ import division
import numpy as np
import pdb
import cv2
import trimesh
import operator
import IPython
import math
import time

# Debug
from line_profiler import LineProfiler

# Point cloud library imports
import pcl
import pcl.pcl_visualization

# Autolab imports
from autolab_core import PointCloud, RigidTransform
from perception import DepthImage, CameraIntrinsics, RenderMode
from visualization import Visualizer2D as vis2d, Visualizer3D as vis3d
from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera

from visualize_placements import *

ci_file = '/nfs/diskstation/projects/dex-net/placing/datasets/sim/sim_06_22/image_dataset/depth_ims/intrinsics_000002.intr'
ci = CameraIntrinsics.load(ci_file)

cp_file = '/nfs/diskstation/projects/dex-net/placing/datasets/sim/sim_06_22/image_dataset/depth_ims/pose_000002.tf'
cp = RigidTransform.load(cp_file)

img_file = '/nfs/diskstation/projects/dex-net/placing/datasets/sim/sim_06_22/image_dataset/depth_ims/image_000002.npy'
mesh_file = 'demon_helmet.obj'

indices, model, image, pc = largest_planar_surface(img_file)
mesh, best_pose, rt = find_stable_poses(mesh_file)
shadow = find_shadow(mesh, best_pose, model[0:3])

def grid_search(pc, indices, model, shadow, img_file):
    length, width, height = shadow.extents
    split_size = max(length, width)
    pc_data, ind = get_pc_data(pc)
    maxes = np.max(pc_data, axis=0)
    mins = np.min(pc_data, axis=0)
    bin_base = mins[2]
    plane_normal = model[0:3]

    scores = np.zeros((int(np.round((maxes[0]-mins[0])/split_size)), int(np.round((maxes[1]-mins[1])/split_size))))
    for i in range(int(np.round((maxes[0]-mins[0])/split_size))):
        x = mins[0] + i*split_size
        for j in range(int(np.round((maxes[1]-mins[1])/split_size))):
            y = mins[1] + j*split_size
            for sh in rotations(shadow, 8):
                scores[i][j] = do_stuff(pc, indices, model, sh, img_file)


    print("\nScores: \n" + str(scores))
    best = best_cell(scores)
    print("\nBest Cell: " + str(best) + ", with score = " + str(scores[best[0]][best[1]]))
    #-------
    # Visualize best placement
    vis3d.figure()
    x = mins[0] + best[0]*split_size
    y = mins[1] + best[1]*split_size
    cell_indices = np.where((x < pc_data[:,0]) & (pc_data[:,0] < x+split_size) & (y < pc_data[:,1]) & (pc_data[:,1] < y+split_size))[0]
    points = pc_data[cell_indices]
    rest = pc_data[np.setdiff1d(np.arange(len(pc_data)), cell_indices)]
    vis3d.points(points, color=(0,1,1))

def do_stuff(pc, indices, model, rotated_shadow, img_file):
    scene = Scene()
    camera = VirtualCamera(ci, cp)
    scene.camera = camera

    # Works
    shadow_obj = SceneObject(rotated_shadow)
    scene.add_object('shadow', shadow_obj)
    wd = scene.wrapped_render([RenderMode.DEPTH])[0]
    wd_bi = wd.to_binary()
    vis2d.figure()
    vis2d.imshow(wd_bi)
    vis2d.show()
   
    # Doesn't work yet
    plane = pc.data.T[indices]
    plane_pc = PointCloud(plane.T, pc.frame)
    di = ci.project_to_image(plane_pc)
    bi = di.to_binary()
    vis2d.figure()
    vis2d.imshow(bi)
    vis2d.show()

    # Works
    both = bi.mask_binary(wd_bi)
    vis2d.figure()
    vis2d.imshow(both)
    vis2d.show()



"""
    # Find a binary mask for the clutter in the bin
    clutter_indices, clutter_mask = find_clutter(pc, indices, model)
    bi = depth_to_bin(img_file)
    # Find a binary mask for the shadow in the plane of the bin
    scene = Scene()
    # Create a VirtualCamera
    camera = VirtualCamera(ci, cp)
    # Add the camera and shadow to the scene
    scene.camera = camera
    shadow_obj = SceneObject(shadow)
    scene.add_object('shadow', shadow_obj)
    wrapped_segmask = scene.wrapped_render([RenderMode.SEGMASK]) # This is a list
    segmask = wrapped_segmask[0] # binary img
    # AND the two masks
    print("BI data: " + str(bi.data))
    print("Segmask: " + str(segmask.data))
    print("Nonzero in segmask = " + str(np.count_nonzero(segmask)))
    both = np.logical_and(bi.data, segmask.data)
    # Compute and return a score
    return np.count_nonzero(both)
"""

grid_search(pc, indices, model, shadow, img_file)
