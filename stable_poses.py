import numpy as np
import trimesh
import operator

from visualization import Visualizer2D as vis2d, Visualizer3D as vis3d

# Load the mesh and compute stable poses
bc = trimesh.load('bar_clamp.obj')
stable_poses, probs = bc.compute_stable_poses()
assert len(stable_poses) == len(probs)

# Find the most probable stable pose
i, value = max(enumerate(probs), key=operator.itemgetter(1))
best_pose = stable_poses[i]
print("Most probable pose:")
print(best_pose)

# Visualize the mesh
# TODO: visualize mesh in the best stable pose (turn pose into an autolab_core.RigidTransform)
vis3d.mesh(bc)
vis3d.show()
