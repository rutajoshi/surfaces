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

""" 3. Given the object in the stable pose, find the shadow of the convex hull. This is the bottom faces of the hull. """
def find_shadow(mesh, best_pose, plane_normal):
    print("Plane normal = " + str(plane_normal))
    mesh = mesh.apply_transform(best_pose)
    ch = mesh.convex_hull
    faces_to_keep, fn_to_keep = [], []
    for face, face_normal in zip(ch.faces, ch.face_normals):
        dot_prod = np.dot(face_normal, plane_normal)
        if dot_prod > 0:
            faces_to_keep.append(face)
            fn_to_keep.append(face_normal)
    shadow = trimesh.Trimesh(vertices=ch.vertices, faces=faces_to_keep, face_normals=fn_to_keep)
    shadow.remove_unreferenced_vertices()
    return shadow

def get_pc_data(pc):
    pc_data = pc.data.T
    all_indices = np.where((pc_data[::,1] < 0.16) & (pc_data[::,1] > -0.24) & (pc_data[::,0] > -0.3) & (pc_data[::,0] < 0.24))[0]
    pc_data = pc_data[all_indices]
    #pc_data = pc_data[np.where(pc_data[::,1] < 0.16)] # remove the empty space before the start of the bin 
    #pc_data = pc_data[np.where(pc_data[::,1] > -0.24)]
    #pc_data = pc_data[np.where(pc_data[::,0] > -0.3)]
    #pc_data = pc_data[np.where(pc_data[::,0] < 0.24)]
    return pc_data, all_indices

""" 4. Given cell extrema, find all points that are under the shadow """
def find_intersections(pc, minx, miny, maxx, maxy, shadow, plane_normal):
    pc_data, ind = get_pc_data(pc)
    cell_indices = np.where((minx < pc_data[:,0]) & (pc_data[:,0] < maxx) & (miny < pc_data[:,1]) & (pc_data[:,1] < maxy))[0]
    points = pc_data[cell_indices]
    
    cell_centroid = np.array([(minx+maxx)/2, (miny+maxy)/2, np.mean(pc_data[::,2])])
    mesh_centroid = shadow.centroid
    translation = cell_centroid - mesh_centroid
    shadow.apply_translation(translation)

    ray = np.array([plane_normal[0], plane_normal[1], -1*plane_normal[2]])
    ray_tracing = shadow.ray.intersects_any(points, np.tile(ray, (len(points), 1)))
    #print("Intersection pts = " + str(np.count_nonzero(ray_tracing)))
    pts_in_shadow = points[ray_tracing]
    shadow_indices = cell_indices[ray_tracing]
    real_pts = pc_data[shadow_indices]
    #print("pts_in_shadow = " + str(pts_in_shadow))
    #print("real_pts = " + str(real_pts))
    # Visualize
    pts_in_shadow_pc = PointCloud(pts_in_shadow.T, pc.frame)
    di = ci.project_to_image(pts_in_shadow_pc)


    #ray_tracing = shadow.ray.intersects_any(pc_data, np.tile(ray, (len(pc_data), 1)))
    #pts_in_shadow = pc_data[ray_tracing]
    #shadow_indices = np.where(ray_tracing==True)[0]

    #vis3d.figure()
    #vis3d.mesh(shadow)
    #display_pc = ci.deproject(di)
    #vis3d.points(display_pc, color=(0,1,1))
    #vis3d.show()
    
    #vis2d.figure()
    #vis2d.imshow(di)
    #vis2d.show()
    # --
    return pts_in_shadow, shadow_indices

""" Generate n rotations for the given shadow, equally spaced around the unit circle. """
def rotations(shadow, n):
    theta = (360 / n) * (2 * math.pi / 360) 
    r = np.array([[math.cos(theta), -1*math.sin(theta), 0, 0],
                      [math.sin(theta), math.cos(theta), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0]])
    rotated_shadows = [shadow]
    for i in range(1,n):
        shadow.apply_transform(r)
        rotated_shadows.append(shadow)
    return rotated_shadows

""" Find clutter: everything not belonging to the plane found by segmentation """
def find_clutter(pc, indices, model):
    not_in_plane_indices = np.setdiff1d(np.arange(len(pc.data.T)), indices) # indexes into pc.data.T
    pc_data, in_bin_indices = get_pc_data(pc) # indexes into pc.data.T
    clutter_in_bin_indices = np.intersect1d(not_in_plane_indices, in_bin_indices) # indexes into pc.data.T
    new_indices_mask = np.isin(in_bin_indices, clutter_in_bin_indices)
    clutter_indices = np.where(new_indices_mask==True)[0]
    return clutter_indices

""" Visualize in 3d, clutter vs non-clutter points """
def binarized_clutter_image(pc, indices, model):
    pc_data, ind = get_pc_data(pc)
    plane = pc.data.T[indices]
    clutter_indices = find_clutter(pc, indices, model)
    clutter = pc_data[clutter_indices]
    #vis3d.figure()
    #vis3d.points(plane, color=(1,0,0))
    #vis3d.points(clutter, color=(0,1,0))
    #vis3d.show()
    return plane, clutter, clutter_indices

""" Visualize in 3d, shadow vs non-shadow points """
def binarized_shadow_image(pc, minx, miny, maxx, maxy, shadow, plane_normal):
    pc_data, ind = get_pc_data(pc)
    pts_in_shadow, shadow_indices = find_intersections(pc, minx, miny, maxx, maxy, shadow, plane_normal)
    not_in_shadow = np.setdiff1d(np.arange(len(pc_data)), shadow_indices)
    not_in_shadow = pc_data[not_in_shadow]
    #print(not_in_shadow.shape)
    #vis3d.figure()
    #vis3d.points(pts_in_shadow, color=(1,0,0))
    #vis3d.points(not_in_shadow, color=(0,1,0))
    #vis3d.show()
    return not_in_shadow, pts_in_shadow, shadow_indices

""" Visualize in 3d overlap between shadow and clutter """
def binarized_overlap_image(pc, minx, miny, maxx, maxy, shadow, plane_normal, indices, model):
    pc_data, ind = get_pc_data(pc)
    plane, clutter, clutter_indices = binarized_clutter_image(pc, indices, model)
    not_in_shadow, pts_in_shadow, shadow_indices = binarized_shadow_image(pc, minx, miny, maxx, maxy, shadow, plane_normal)
    overlap_indices = np.intersect1d(clutter_indices, shadow_indices)
    #print("Clutter = " + str(clutter_indices))
    #print("Shadow = " + str(shadow_indices))
    #print("Number of overlap indices = " + str(len(overlap_indices)))
    #print("Overlap indices = " + str(overlap_indices))
    overlap = pc_data[overlap_indices]
    rest = pc_data[np.setdiff1d(np.arange(len(pc_data)), overlap_indices)]
    
    #vis3d.figure()
    #vis3d.points(pc_data[shadow_indices], color=(0,0,1)) # shadow
    #vis3d.points(clutter, color=(1,0,0)) # clutter
    #vis3d.show()

    #vis3d.figure()
    #vis3d.points(rest, color=(1,0,1)) # non-overlapping
    #vis3d.points(overlap, color=(0,1,0)) #overlapping shadow and clutter
    #vis3d.show()

    return len(overlap)


""" 5. Go through the cells of a given bin image """
def grid_search(pc, indices, model, shadow):
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

            #binarized_overlap_image(pc, x, y, x+split_size, y+split_size, shadow, plane_normal, indices, model)

            for sh in rotations(shadow, 8):
                overlap_size = binarized_overlap_image(pc, x, y, x+split_size, y+split_size, shadow, plane_normal, indices, model)
                #pts_in_shadow, shadow_indices = find_intersections(pc, x, y, x+split_size, y+split_size, sh, plane_normal)
                scores[i][j] = -1*overlap_size


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
    vis3d.points(rest, color=(1,0,1))
    vis3d.show()
    #--------

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
    
    #vis3d.figure()
    #vis3d.points(pc, color=(1,0,0))

    #vis3d.mesh(shadow, rt)
    #vis3d.show()

    grid_search(pc, indices, model, shadow)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__": main()
