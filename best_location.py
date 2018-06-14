# General imports
import numpy as np
import pdb
import cv2
import trimesh
import operator
import IPython
import math

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

""" 4. Score each cell given the footprint and other metrics """
def score_cells(pc, indices, model, shadow):
    length, width, height = shadow.extents
    split_size = max(length, width)
    # go through all the cells, assuming that the mesh is on the same scale as the bin
    # TODO: finish this
    pc_plane = pc.data.T[indices]
    pc_plane = pc_plane[np.where(pc_plane[::,1] < 0.16)] # remove the empty space before the start of the bin (applies only to the yumi)
    pc_all = pc.data.T
    maxes = np.max(pc_plane, axis=0)
    mins = np.min(pc_plane, axis=0)
    bin_base = (mins[2]+maxes[2])/2
    scores = np.zeros((int(np.round((maxes[0]-mins[0])/split_size)), int(np.round((maxes[1]-mins[1])/split_size))))
    scores_all = np.zeros((int(np.round((maxes[0]-mins[0])/split_size)), int(np.round((maxes[1]-mins[1])/split_size))))
    scores_dp = np.zeros((int(np.round((maxes[0]-mins[0])/split_size)), int(np.round((maxes[1]-mins[1])/split_size))))
    # Compute score for each cell
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

            # find the number of points in the pc data that are within the cell and the plane
            pc_cell_plane = pc_plane[x < pc_plane[:,0]]
            pc_cell_plane = pc_cell_plane[pc_cell_plane[:,0] < (x + split_size)]
            pc_cell_plane = pc_cell_plane[y < pc_cell_plane[:,1]]
            pc_cell_plane = pc_cell_plane[pc_cell_plane[:,1] < (y + split_size)]

            # find the number of points that are in the 3d cell
            pc_cell = pc_all[x < pc_all[:,0]]
            pc_cell = pc_cell[pc_cell[:,0] < (x + split_size)]
            pc_cell = pc_cell[y < pc_cell[:,1]]
            pc_cell = pc_cell[pc_cell[:,1] < (y + split_size)]

            # retrieve a score based on number of points in the cell and plane within shadow extents
            intersecting_pts, score_planei = rotational_high_score(8, score_plane_intersection, pc_cell_plane, cell_centroid, shadow)
            scores[i][j] = score_planei
            #vis3d.points(pc_plane, color=(0,1,0))
            #vis3d.points(intersecting_pts, color=(1,0,0))
            #vis3d.mesh(shadow)
            #vis3d.show()

            intersecting_pts, score_totali = rotational_high_score(8, score_total_intersection, pc_cell, cell_centroid, shadow)
            scores_all[i][j] = score_totali
            #vis3d.points(pc_all, color=(0,1,0))
            #vis3d.points(intersecting_pts, color=(1,0,0))
            #vis3d.mesh(shadow)
            #vis3d.show()

            indices, score_dp = rotational_high_score(8, score_cell_planefit, pc_cell, cell_centroid, shadow)
            scores_dp[i][j] = score_dp

            # un-translate the mesh before the next iteration
            shadow.apply_translation(untranslation)
    print("\nScores [plane]: \n" + str(scores))
    best_plane = best_cell(scores)
    print("\nBest cell [plane] = " + str(best_plane))
    print("\nScores [total]: \n" + str(scores_all))
    best_all = best_cell(scores_all)
    print("\nBest cell [total] = " + str(best_all))
    print("\nScores [dot product]: \n" + str(scores_dp))
    best_dp = best_cell(scores_dp)
    print("\nBest cell [dot product] = " + str(best_dp))
    return scores, split_size

def rotational_high_score(n, scoring_function, pc_cell, cell_centroid, shadow):
    rotated_shadows = rotations(shadow, n)
    intersecting_pts_arrays = []
    scores = []
    for sh in rotated_shadows:
        ip, score = scoring_function(pc_cell, cell_centroid, sh)
        scores.append(score)
        intersecting_pts_arrays.append(ip)
    i = np.argmax(scores)
    return intersecting_pts_arrays[i], scores[i]

def score_plane_intersection(cell_pc, cell_centroid, shadow):
    """
    The score of this cell is the number of points that intersect with the plane.
    Assumes that the cell_pc is the set of all points in both the plane and the cell.
    
    The score is higher if more points intersect (since clutter and box edges are not in the flat plane, they don't intersect).
    """
    length, width, height = shadow.extents
    minx, maxx = cell_centroid[0] - (length / 2), cell_centroid[0] + (length / 2)
    miny, maxy = cell_centroid[1] - (width / 2), cell_centroid[1] + (width / 2)
    intersecting_pts = cell_pc[minx < cell_pc[:,0]]
    intersecting_pts = intersecting_pts[intersecting_pts[:,0] < maxx]
    intersecting_pts = intersecting_pts[miny < intersecting_pts[:,1]]
    intersecting_pts = intersecting_pts[intersecting_pts[:,1] < maxy]
    return intersecting_pts, len(intersecting_pts)

def score_total_intersection(cell_pc, cell_centroid, shadow):
    """
    The score of this cell is the number of points that intersect.
    Assumes that the cell_pc is the cell of the full point cloud.

    The score is higher if fewer points intersect (since clutter and box edges intersect with the object as well.
    Return -1*len(intersecting points)
    """
    length, width, height = shadow.extents
    minx, maxx = cell_centroid[0] - (length / 2), cell_centroid[0] + (length / 2)
    miny, maxy = cell_centroid[1] - (width / 2), cell_centroid[1] + (width / 2)
    intersecting_pts = cell_pc[minx < cell_pc[:,0]]
    intersecting_pts = intersecting_pts[intersecting_pts[:,0] < maxx]
    intersecting_pts = intersecting_pts[miny < intersecting_pts[:,1]]
    intersecting_pts = intersecting_pts[intersecting_pts[:,1] < maxy]
    return intersecting_pts, -1*len(intersecting_pts)

""" Score the cell using the points in that cell from the planar surface that intersect with the shadow """
def score_cell_planefit(cell_pc, cell_centroid, shadow):
    intersecting_pts, len_of_those = score_total_intersection(cell_pc, cell_centroid, shadow)
    # Fit a plane to the points in the cell that intersect with the shadow
    # Make a PCL type point cloud from the image
    p = pcl.PointCloud(intersecting_pts.astype(np.float32))
    # Make a segmenter and segment the point cloud.
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PARALLEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.005)
    indices, model = seg.segment()
    
    # score = dot product of plane normal to face normal of the shadow
    fn = shadow.face_normals[3] # TODO: figure out which normal is the right one to use
    score = np.dot(fn, model[0:3])
    return indices, score

""" 5. Return the cell with the highest score """
def best_cell(scores):
    ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    return ind # tuple

""" Main """
def main():
    img_file = '/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/depth_ims_numpy/image_000001.npy'
    ci_file = '/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/camera_intrinsics.intr'
    mesh_file = 'demon_helmet.obj'

    indices, model, image, pc = largest_planar_surface(img_file, ci_file)
    mesh, best_pose, rt = find_stable_poses(mesh_file)
    shadow = find_shadow(mesh, best_pose, model[0:3])

    #vis3d.mesh(shadow, rt)
    #vis3d.show()

    scores, split_size = score_cells(pc, indices, model, shadow)
    ind = best_cell(scores)
    # print("Scores: \n" + str(scores))
    # print("\nBest cell = " + str(ind))


if __name__ == "__main__": main()

