import numpy as np
import trimesh
import operator

from autolab_core import RigidTransform
from visualization import Visualizer2D as vis2d, Visualizer3D as vis3d

# Load the mesh and compute stable poses
bc = trimesh.load('demon_helmet.obj')
stable_poses, probs = bc.compute_stable_poses()
assert len(stable_poses) == len(probs)

# Find the most probable stable pose
i, value = max(enumerate(probs), key=operator.itemgetter(1))
best_pose = stable_poses[i]
print("Most probable pose:")
print(best_pose)
print("Probability = " + str(probs[i]))

# Visualize the mesh in the most probable stable state
rotation = np.asarray(best_pose[:3, :3])
translation = np.asarray(best_pose[:3, 3])
rt = RigidTransform(rotation, translation, from_frame='obj', to_frame='world')
vis3d.mesh(bc, rt)
vis3d.show()

# TODO: Project the mesh onto the plane defined by the largest flat surface in the bin
