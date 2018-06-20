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

# Globals
ci_file = '/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/camera_intrinsics.intr'
ci = CameraIntrinsics.load(ci_file)

""" 1. Given depth image of bin, retrieve largest planar surface """
def largest_planar_surface(filename):
    # Load the image as a numpy array and the camera intrinsics
    image = np.load(filename)
    # Create and deproject a depth image of the data using the camera intrinsics
    di = DepthImage(image, frame=ci.frame)
    di = di.inpaint()
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

""" 2. Given an object mesh, find stable poses """
def find_stable_poses(mesh_file):
    # Load the mesh and compute stable poses
    mesh = trimesh.load(mesh_file)
    stable_poses, probs = mesh.compute_stable_poses()
    assert len(stable_poses) == len(probs)
    # Find the most probable stable pose
    i, value = max(enumerate(probs), key=operator.itemgetter(1))
    best_pose = stable_poses[i]
    # Visualize the mesh in the most probable stable state
    rotation = np.asarray(best_pose[:3, :3])
    translation = np.asarray(best_pose[:3, 3])
    rt = RigidTransform(rotation, translation, from_frame='obj', to_frame='world')
    # vis3d.mesh(mesh, rt)
    # vis3d.show()
    return mesh, best_pose, rt

""" 3. Given the object in the stable pose, find the shadow of the convex hull """
def find_shadow(mesh, best_pose, plane_normal):
    print("Plane normal = " + str(plane_normal))
    mesh = mesh.apply_transform(best_pose)
    ch = mesh.convex_hull
    faces_to_keep, fn_to_keep = [], []
    for face, face_normal in zip(ch.faces, ch.face_normals):
        dot_prod = np.dot(face_normal, plane_normal)
        if dot_prod < 0:
            faces_to_keep.append(face)
            fn_to_keep.append(face_normal)
    shadow = trimesh.Trimesh(vertices=ch.vertices, faces=faces_to_keep, face_normals=fn_to_keep)
    shadow.remove_unreferenced_vertices()
    return shadow

""" 4. Given cell extrema, find all points that are under the shadow """
def find_intersections(pc, minx, miny, maxx, maxy, shadow, plane_normal):
    pc_data = pc.data.T
    points = pc_data[minx < pc_data[:,0]]
    points = points[points[:,0] < maxx]
    points = points[miny < points[:,1]]
    points = points[points[:,1] < maxy]
    ray_tracing = shadow.ray.intersects_any(points, np.tile(plane_normal, (len(points), 1)))
    pts_in_shadow = points[ray_tracing]
    # Visualize
    pts_in_shadow_pc = PointCloud(pts_in_shadow.T, pc.frame)
    di = ci.project_to_image(pts_in_shadow_pc)
    #vis2d.figure()
    #vis2d.imshow(di)
    #vis2d.show()
    # --
    return pts_in_shadow

""" 5. Go through the cells of a given bin image """
def grid_search(pc, indices, model, shadow):
    length, width, height = shadow.extents
    split_size = max(length, width)
    pc_data = pc.data.T
    pc_data = pc_data[np.where(pc_data[::,1] < 0.16)] # remove the empty space before the start of the bin 
    maxes = np.max(pc_data, axis=0)
    mins = np.min(pc_data, axis=0)
    bin_base = mins[2]
    plane_normal = model[0:3]

    scores = np.zeros((int(np.round((maxes[0]-mins[0])/split_size)), int(np.round((maxes[1]-mins[1])/split_size))))
    for i in range(int(np.round((maxes[0]-mins[0])/split_size))):
        x = mins[0] + i*split_size
        for j in range(int(np.round((maxes[1]-mins[1])/split_size))):
            y = mins[1] + j*split_size
            # translate the mesh to the center of the cell
            mesh_centroid = shadow.centroid
            cell_centroid = np.array([x + (split_size / 2), y + (split_size / 2), bin_base])
            translation = cell_centroid - mesh_centroid
            untranslation = -1 * translation
            shadow.apply_translation(translation)

            pts_in_shadow = find_intersections(pc, x, y, x+split_size, y+split_size, shadow, plane_normal)
            scores[i][j] = len(pts_in_shadow)

            # un-translate the mesh before the next iteration
            shadow.apply_translation(untranslation)

    print("\nScores: \n" + str(scores))
    best = best_cell(scores)

""" 6. Return the cell with the highest score """
def best_cell(scores):
    ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    return ind # tuple

def main():
    start_time = time.time()
    img_file = '/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/depth_ims_numpy/image_000001.npy'
    mesh_file = 'demon_helmet.obj'

    indices, model, image, pc = largest_planar_surface(img_file)
    mesh, best_pose, rt = find_stable_poses(mesh_file)
    shadow = find_shadow(mesh, best_pose, model[0:3])
    
    vis3d.figure()
    vis3d.points(pc, color=(1,0,0))

    vis3d.mesh(shadow, rt)
    vis3d.show()

    grid_search(pc, indices, model, shadow)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__": main()
