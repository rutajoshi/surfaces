import numpy as np
import pcl

from autolab_core import PointCloud
from perception import DepthImage
from visualization import Visualizer3D as vis3d
from visualization import Visualizer2D as vis2d

arr = np.load('/nfs/diskstation/projects/dex-net/placing/datasets/real/sample_ims_05_22/depth_ims_numpy/image_000000.npy')
di = DepthImage(arr)
image = di._data
vis2d.imshow(di)

p = pcl.PointCloud()
p.from_array(image)

seg = self.p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
indices, model = seg.segment()


