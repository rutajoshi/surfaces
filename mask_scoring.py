# Consolidating all the bad code into something potentially more understandable

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

""" 
@author: Ruta Joshi
"""

# Globals
ci_file = '/nfs/diskstation/projects/dex-net/placing/datasets/sim/sim_06_22/image_dataset/depth_ims/intrinsics_000002.intr'
ci = CameraIntrinsics.load(ci_file)
cp_file = '/nfs/diskstation/projects/dex-net/placing/datasets/sim/sim_06_22/image_dataset/depth_ims/pose_000002.tf'
cp = RigidTransform.load(cp_file)
#print("Camera pose:\n" + str(cp))

""" 1. Given depth image of bin, retrieve largest planar surface """
#@profile
def largest_planar_surface(filename):
    # Load the image as a numpy array and the camera intrinsics
    image = np.load(filename)
    # Create and deproject a depth image of the data using the camera intrinsics
    di = DepthImage(image, frame=ci.frame)
    di = di.inpaint()
    pc = ci.deproject(di)
    pc = cp.apply(pc)
    # Make a PCL type point cloud from the image
    p = pcl.PointCloud(pc.data.T.astype(np.float32))
    # Make a segmenter and segment the point cloud.
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PARALLEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.005)
    indices, model = seg.segment()
    return indices, model, image, pc

#@profile
def largest_planar_surface_di(di, ci, cp):
    di = DepthImage(image, frame=ci.frame)
    di = di.inpaint()
    pc = ci.deproject(di)
    pc = cp.apply(pc)
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
#@profile
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

#@profile
def find_stable_poses_mesh(mesh):
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
    return best_pose, rt

""" 3. Given the object in the stable pose, find the shadow of the convex hull. This is the bottom faces of the hull. """
#@profile
def find_shadow(mesh, best_pose, plane_normal):
    #print("Plane normal = " + str(plane_normal))
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

""" Restrict the data to be only the data within the bounds of the base of the bin (largest planar surface) """
#@profile
def get_pc_data(pc, indices):
    pc_data = pc.data.T
    pc_plane = pc.data.T[indices]
    mins = np.min(pc_plane, axis=0)
    maxes = np.max(pc_plane, axis=0)
    all_indices = np.where((pc_data[::,0] > mins[0]) & (pc_data[::,0] < maxes[0]) & (pc_data[::,1] > mins[1]) & (pc_data[::,1] < maxes[1]))
    pc_data = pc_data[all_indices]
    return pc_data, all_indices

""" Get all the data for points actually on the plane. """
#@profile
def get_plane_data(pc, indices):
    pc_plane = pc.data.T[indices]
    mins = np.min(pc_plane, axis=0)
    maxes = np.max(pc_plane, axis=0)
    all_indices = np.where((pc_plane[::,0] > mins[0]) & (pc_plane[::,0] < maxes[0]) & (pc_plane[::,1] > mins[1]) & (pc_plane[::,1] < maxes[1]))
    pc_plane = pc_plane[all_indices]
    return pc_plane

""" Returns T_obj_world rigid transforms for the n rotations of a shadow translated to a cell. """
def transforms(pc, pc_data, shadow, minx, miny, maxx, maxy, n, original_tow):
    # Find the translation component
    cell_centroid = np.array([(minx+maxx)/2, (miny+maxy)/2, np.mean(pc_data[::,2])])
    mesh_centroid = shadow.centroid
    translation = cell_centroid - mesh_centroid
    #print("Translation = " + str(translation))
    global_translation = original_tow.translation
    #print("Global translation = " + str(global_translation))
    #print("Global rotation = " + str(original_tow.rotation))
    translation = translation + global_translation
    #print("Total translation = " + str(translation))
    #translation = translation + cp.translation
    #print("ACTUAL Total translation = " + str(translation))
    #shadow.apply_translation(translation)

    # Find each rotation component and create a transform
    transforms = []
    theta = (360 / n) * (2 * math.pi / 360)
    for i in range(n):
        angle = i*theta
        rotation = np.array([[math.cos(angle), -1*math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]])
        #rotation = np.eye(3)
        rigid_transform = RigidTransform(rotation=rotation, translation=translation, from_frame=pc.frame, to_frame='world')
        transforms.append(rigid_transform)

    # Return the list of transforms
    return transforms

""" Find points under the shadow by rendering a binary mask of clutter. """
#@profile
def under_shadow(scene, bin_bi):
    wd = scene.wrapped_render([RenderMode.DEPTH])[0]
    wd_bi = wd.to_binary()
    
    #vis2d.figure()
    #vis2d.imshow(wd)
    #vis2d.show()

    #vis2d.figure()
    #vis2d.imshow(wd_bi)
    #vis2d.show()

    #vis2d.figure()
    #vis2d.imshow(bin_bi)
    #vis2d.show()

    both = bin_bi.mask_binary(wd_bi)
    #vis2d.figure()
    #vis2d.imshow(both)
    #vis2d.show()

    score = np.count_nonzero(both.data)
    return -1*score

""" 6. Return the cell with the highest score """
#@profile
def best_cell(scores):
    ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    return ind

""" Faster grid search """
#@profile
def fast_grid_search(pc, indices, model, shadow):
    length, width, height = shadow.extents
    split_size = max(length, width)
    pc_data, ind = get_pc_data(pc, indices)
    maxes = np.max(pc_data, axis=0)
    mins = np.min(pc_data, axis=0)
    bin_base = mins[2]
    plane_normal = model[0:3]

    #di_temp = ci.project_to_image(pc)
    #vis2d.figure()
    #vis2d.imshow(di_temp)
    #vis2d.show()
    #plane_data = pc.data.T[indices]
    #plane_pc = PointCloud(plane_data.T, pc.frame)
    #di = ci.project_to_image(plane_pc)
    #bi = di.to_binary()

    plane_data = get_plane_data(pc, indices)
    plane_pc = PointCloud(plane_data.T, pc.frame)
    #vis3d.figure()
    #vis3d.points(plane_pc)
    #vis3d.show()
    plane_pc = cp.inverse().apply(plane_pc)
    di = ci.project_to_image(plane_pc)
    bi = di.to_binary()
    bi = bi.inverse()
    #vis2d.figure()
    #vis2d.imshow(bi)
    #vis2d.show()

    scene = Scene()
    camera = VirtualCamera(ci, cp)
    scene.camera = camera
    shadow_obj = SceneObject(shadow)
    scene.add_object('shadow', shadow_obj)
    orig_tow = shadow_obj.T_obj_world
    #tr = transforms(pc, pc_data, shadow, mins[0], mins[1], mins[0]+split_size, mins[1]+split_size, 8, orig_tow)
    #shadow_obj.T_obj_world = tr[0]
    wd = scene.wrapped_render([RenderMode.DEPTH])[0]
    wd_bi = wd.to_binary()
    #vis2d.figure()
    #vis2d.imshow(wd_bi)
    #vis2d.show()

    scores = np.zeros((int(np.round((maxes[0]-mins[0])/split_size)), int(np.round((maxes[1]-mins[1])/split_size))))
    for i in range(int(np.round((maxes[0]-mins[0])/split_size))):
        x = mins[0] + i*split_size
        for j in range(int(np.round((maxes[1]-mins[1])/split_size))):
            y = mins[1] + j*split_size

            for tow in transforms(pc, pc_data, shadow, x, y, x+split_size, y+split_size, 8, orig_tow):
                shadow_obj.T_obj_world = tow
                scores[i][j] = under_shadow(scene, bi)
                shadow_obj.T_obj_world = orig_tow

 
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


""" Finer grid search """
def fine_grid_search(pc, indices, model, shadow):
    length, width, height = shadow.extents
    split_size = max(length, width)
    pc_data, ind = get_pc_data(pc, indices)
    maxes = np.max(pc_data, axis=0)
    mins = np.min(pc_data, axis=0)
    bin_base = mins[2]
    plane_normal = model[0:3]
    splits = 3
    step_size = split_size / splits
    
    plane_data = get_plane_data(pc, indices)
    plane_pc = PointCloud(plane_data.T, pc.frame)
    plane_pc = cp.inverse().apply(plane_pc)
    di = ci.project_to_image(plane_pc)
    bi = di.to_binary()
    bi = bi.inverse()

    scene = Scene()
    camera = VirtualCamera(ci, cp)
    scene.camera = camera
    shadow_obj = SceneObject(shadow)
    scene.add_object('shadow', shadow_obj)
    orig_tow = shadow_obj.T_obj_world

    numx = (int(np.round((maxes[0]-mins[0])/split_size)) - 1) * splits + 1
    numy = (int(np.round((maxes[1]-mins[1])/split_size)) - 1) * splits + 1
    scores = np.zeros((numx, numy))
    for i in range(numx):
        x = mins[0] + i*step_size
        for j in range(numy):
            y = mins[1] + j*step_size

            for tow in transforms(pc, pc_data, shadow, x, y, x+split_size, y+split_size, 8, orig_tow):
                shadow_obj.T_obj_world = tow
                scores[i][j] = under_shadow(scene, bi)
                shadow_obj.T_obj_world = orig_tow

    print("\nScores: \n" + str(scores))
    best = best_cell(scores)
    print("\nBest Cell: " + str(best) + ", with score = " + str(scores[best[0]][best[1]]))
    #-------
    # Visualize best placement
    vis3d.figure()
    x = mins[0] + best[0]*step_size
    y = mins[1] + best[1]*step_size
    cell_indices = np.where((x < pc_data[:,0]) & (pc_data[:,0] < x+split_size) & (y < pc_data[:,1]) & (pc_data[:,1] < y+split_size))[0]
    points = pc_data[cell_indices]
    rest = pc_data[np.setdiff1d(np.arange(len(pc_data)), cell_indices)]
    vis3d.points(points, color=(0,1,1))
    vis3d.points(rest, color=(1,0,1))
    vis3d.show()
    #--------
    return best



######## Main #############
def main():
    start_time = time.time()
    img_file = '/nfs/diskstation/projects/dex-net/placing/datasets/sim/sim_06_22/image_dataset/depth_ims/image_000002.npy'
    mesh_file = 'demon_helmet.obj'

    indices, model, image, pc = largest_planar_surface(img_file)
    mesh, best_pose, rt = find_stable_poses(mesh_file)
    shadow = find_shadow(mesh, best_pose, model[0:3])

    #vis3d.figure()
    #vis3d.points(pc, color=(1,0,0))
    #vis3d.mesh(shadow, rt)
    #vis3d.show()

    fine_grid_search(pc, indices, model, shadow)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__": main()
