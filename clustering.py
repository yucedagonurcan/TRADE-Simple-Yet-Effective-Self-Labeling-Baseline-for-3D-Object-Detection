import numpy as np
import open3d as o3d
from jsk_recognition_msgs.msg import BoundingBox
import shapely

class AdaptiveClustering:
    def __init__(self, start_epsilon, max_epsilon, step_epsilon, min_points, iou_thresh):
        self.start_epsilon = start_epsilon
        self.step_epsilon = step_epsilon
        self.max_epsilon = max_epsilon
        self.min_points = min_points
        self.iou_thresh = iou_thresh

    # Given a point cloud and a template object, this function returns a list of clusters
    #   that can best represent the template object from the point cloud.
    def clustering(self, proposed_pointcloud: o3d.geometry.PointCloud, template_object:BoundingBox=None):

        if proposed_pointcloud is None or proposed_pointcloud.is_empty():
            return None


        # Zero out the z axis of the point cloud
        for i in range(len(proposed_pointcloud.points)):
            proposed_pointcloud.points[i][2] = 0.0

        target_area = template_object.dimensions.x * template_object.dimensions.y

        current_epsilon = self.start_epsilon

        best_points = None
        best_area_diff = float('inf')

        while current_epsilon <= self.max_epsilon:
            current_cluster = proposed_pointcloud.cluster_dbscan(eps=current_epsilon,
                                                                 min_points=self.min_points,
                                                                 print_progress=False)
            cluster_np = np.asarray(current_cluster)
            cluster_indices, cluster_counts = np.unique(cluster_np[cluster_np!=-1], return_counts=True)

            # If there are no clusters, increase the epsilon and continue to the next iteration
            if len(cluster_indices) == 0:
                current_epsilon = current_epsilon + self.step_epsilon
                continue

            biggest_cluster_idx = cluster_indices[np.argmax(cluster_counts)]

            # Get the biggest cluster
            biggest_cluster = proposed_pointcloud.select_by_index(np.where(cluster_np == biggest_cluster_idx)[0])


            # Calculate the area of the biggest cluster
            biggest_cluster_area = shapely.convex_hull(shapely.MultiPoint(np.asarray(biggest_cluster.points)[:,:2])).area

            # Calculate the difference between the area of the biggest cluster and the target area
            area_diff = abs(biggest_cluster_area - target_area)
            if area_diff < best_area_diff:
                best_area_diff = area_diff
                best_points = np.where(cluster_np == biggest_cluster_idx)[0]

            # If we already clustered the entire point cloud, break the loop
            if len(best_points) == len(proposed_pointcloud.points):
                break

            current_epsilon = current_epsilon + self.step_epsilon

        return best_points

def get_cluster(self):
        return None


