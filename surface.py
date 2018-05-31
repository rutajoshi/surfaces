from planar_seg import *

class Surface(Object):
    def __init__(self, point_cloud, parent_cloud):
        self.point_cloud = point_cloud
        self.parent_cloud = parent_cloud
        self.size = len(self.point_cloud)
        self.normal, self.defining_point = self.point_cloud.best_fit_plane()

    def score(self):
        size_ratio = self.size / len(self.parent_cloud)
        return size_ratio
