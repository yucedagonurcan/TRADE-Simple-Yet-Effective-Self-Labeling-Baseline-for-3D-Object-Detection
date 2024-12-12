from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import rospy
from sensor_msgs.msg import PointCloud2

import ros_numpy as rnp
import open3d as o3d
from tf2_ros import Buffer, TransformListener
import numpy as np
from scipy.spatial.transform import Rotation

from clustering import AdaptiveClustering
from l_shape_fitting import LShapeFitting, RectangleData
import time
from shapely.geometry import Polygon

from shapely.affinity import rotate, translate

from scipy.spatial.transform import Rotation as R


MAX_DIST_ASSOCIATION = 5.0
def boxoverlap(a:BoundingBox, b:BoundingBox):

    def create_obb(cx, cy, w, h, theta):
        # Define the initial rectangle
        rectangle = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
        # Rotate and translate the rectangle
        rotated_rect = rotate(rectangle, theta, use_radians=True)
        obb = translate(rotated_rect, cx, cy)
        return obb

    theta_1 = R.from_quat([a.pose.orientation.x,
                           a.pose.orientation.y,
                           a.pose.orientation.z,
                           a.pose.orientation.w]).as_euler('zyx')[0]
    theta_2 = R.from_quat([b.pose.orientation.x,
                           b.pose.orientation.y,
                           b.pose.orientation.z,
                           b.pose.orientation.w]).as_euler('zyx')[0]
    # Define the OBBs
    obb1 = create_obb(a.pose.position.x, a.pose.position.y, a.dimensions.x, a.dimensions.y, theta_1)
    obb2 = create_obb(b.pose.position.x, b.pose.position.y, b.dimensions.x, b.dimensions.y, theta_2)

    # Calculate the intersection area
    intersection_area = obb1.intersection(obb2).area

    # Calculate the union area
    union_area = obb1.area + obb2.area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou



def transform_object_list(objects: BoundingBoxArray, transform_stamped):
    new_objects = BoundingBoxArray()
    new_objects.header.frame_id = transform_stamped.header.frame_id
    new_objects.header.stamp = objects.header.stamp

    transform_mat = np.eye(4)
    transform_mat[0:3, 3] = np.array([transform_stamped.transform.translation.x, transform_stamped.transform.translation.y, transform_stamped.transform.translation.z])
    rot = Rotation.from_quat([transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w])
    transform_mat[0:3, 0:3] = rot.as_matrix()


    for obj in objects.boxes:
        new_obj = BoundingBox()
        new_obj.header.stamp = transform_stamped.header.stamp
        new_obj.header.frame_id = transform_stamped.header.frame_id

        object_transformation = np.eye(4)
        object_transformation[0:3, 3] = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
        rot = Rotation.from_quat([obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w])
        object_transformation[0:3, 0:3] = rot.as_matrix()

        transformed_pose = np.dot(transform_mat, object_transformation)

        new_obj.pose.position.x = transformed_pose[0, 3]
        new_obj.pose.position.y = transformed_pose[1, 3]
        new_obj.pose.position.z = transformed_pose[2, 3]

        rot = Rotation.from_matrix(transformed_pose[0:3, 0:3])
        quat = rot.as_quat()
        new_obj.pose.orientation.x = quat[0]
        new_obj.pose.orientation.y = quat[1]
        new_obj.pose.orientation.z = quat[2]
        new_obj.pose.orientation.w = quat[3]

        new_obj.dimensions.x = obj.dimensions.x
        new_obj.dimensions.y = obj.dimensions.y
        new_obj.dimensions.z = obj.dimensions.z

        new_obj.label = obj.label
        new_obj.value = obj.value

        new_objects.boxes.append(new_obj)

    return new_objects

def extend_object_list(objects: BoundingBoxArray, extension_factor):
    new_objects = BoundingBoxArray()
    new_objects.header = objects.header

    for obj in objects.boxes:
        new_obj = BoundingBox()
        new_obj.header = obj.header

        new_obj.pose = obj.pose
        new_obj.dimensions = obj.dimensions
        new_obj.label = obj.label
        new_obj.value = obj.value

        new_obj.dimensions.x *= extension_factor
        new_obj.dimensions.y *= extension_factor
        new_obj.dimensions.z *= extension_factor + 1.5

        new_objects.boxes.append(new_obj)

    return new_objects

class TrackerGuidedDetector:
    def __init__(self, tracked_objects_topic, pointcloud_topic, output_topic):
        self.track_sub = rospy.Subscriber(tracked_objects_topic, BoundingBoxArray, self.track_callback, queue_size=10)
        self.cloud_sub =  rospy.Subscriber(pointcloud_topic, PointCloud2, self.cloud_callback, queue_size=10)

        self.finetuned_tracks_pub = rospy.Publisher(output_topic, BoundingBoxArray, queue_size=10, latch=True)
        self.debug_cloud_pub = rospy.Publisher('~debug_cloud/cropped', PointCloud2, queue_size=10, latch=True)
        self.debug_cloud_cluster_pub = rospy.Publisher('~debug_cloud/valid_cluster', PointCloud2, queue_size=1, latch=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.extension_factor = 1.5

        self.tracklets = None

        self.adaptive_clustering = AdaptiveClustering(0.3, 1.6, 0.3, 5, 0.5)
        self.l_shape_fitting = LShapeFitting()

        self.cloud_ground_height_threshold = 0.2
        self.lidar_height = 2.4

        self.iou_threshold = 0.3


    def get_transform(self, target_frame, source_frame, time):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
            return transform
        except Exception as e:
            print(e)
            return None

    def track_callback(self, tracks_msg):
        self.tracklets = tracks_msg

    def predict_tracklets(self, dt):
        if self.tracklets is None:
            return None

        predicted_tracklets = BoundingBoxArray()
        predicted_tracklets.header = self.tracklets.header
        predicted_tracklets.header.stamp = self.tracklets.header.stamp + rospy.Duration(dt)

        for trk in self.tracklets.boxes:

            track_position = np.array([trk.pose.position.x, trk.pose.position.y, trk.pose.position.z])
            track_rot  = Rotation.from_quat([trk.pose.orientation.x, trk.pose.orientation.y, trk.pose.orientation.z, trk.pose.orientation.w])

            track_velocity = trk.value

            track_velocity_shift = np.array([track_velocity, 0, 0])
            track_shift_in_map = np.dot(track_rot.as_matrix(), track_velocity_shift)

            shifted_position = track_position + track_shift_in_map * dt

            new_trk = BoundingBox()
            new_trk.header = self.tracklets.header
            new_trk.pose.position.x = shifted_position[0]
            new_trk.pose.position.y = shifted_position[1]

            new_trk.pose.orientation = trk.pose.orientation
            new_trk.dimensions = trk.dimensions
            new_trk.label = trk.label
            new_trk.value = trk.value

            predicted_tracklets.boxes.append(new_trk)

        return predicted_tracklets

    def cloud_callback(self, cloud_msg):


        start_callback = time.time()

        if self.tracklets is None:
            bbox_arr = BoundingBoxArray()
            bbox_arr.header = cloud_msg.header
            bbox_arr.boxes = []

            self.finetuned_tracks_pub.publish(bbox_arr)
            return


        cloud_np = rnp.point_cloud2.pointcloud2_to_xyz_array(cloud_msg)
        cloud_np = cloud_np[cloud_np[:, 2] > -self.lidar_height + self.cloud_ground_height_threshold]

        time_diff = cloud_msg.header.stamp.to_sec() - self.tracklets.header.stamp.to_sec()
        predicted_tracks = self.predict_tracklets(time_diff)


        tracks2cloud = self.get_transform(cloud_msg.header.frame_id, predicted_tracks.header.frame_id, predicted_tracks.header.stamp)
        if tracks2cloud is None:
            print("Could not get transform")
            return

        tracks_transformed = transform_object_list(predicted_tracks, tracks2cloud)
        extended_tracks = extend_object_list(tracks_transformed, 1.5)

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_np)


        intensity_vec = np.zeros((len(cloud_np), 1))
        clusters_vec = np.zeros((len(cloud_np), 1))

        fine_tuned_tracks = BoundingBoxArray()
        fine_tuned_tracks.header = cloud_msg.header


        mean_clustering_time = 0
        mean_fitting_time = 0


        for i, trk in enumerate(extended_tracks.boxes):

            trk_transformation = np.eye(4)
            trk_transformation[0:3, 3] = np.array([trk.pose.position.x, trk.pose.position.y, trk.pose.position.z])
            rot = Rotation.from_quat([trk.pose.orientation.x, trk.pose.orientation.y, trk.pose.orientation.z, trk.pose.orientation.w])
            trk_transformation[0:3, 0:3] = rot.as_matrix()

            trk_o3d = o3d.geometry.OrientedBoundingBox()
            trk_o3d.center = trk_transformation[0:3, 3]
            trk_o3d.R = trk_transformation[0:3, 0:3]

            trk_o3d.extent = np.array([trk.dimensions.x, trk.dimensions.y, trk.dimensions.z])

            inlier_indices = trk_o3d.get_point_indices_within_bounding_box(cloud_o3d.points)
            inlier_indices = np.asarray(inlier_indices)

            if len(inlier_indices) < 5:
                continue

            intensity_vec[inlier_indices] = (i+1)*100 + 1

            clustering_start = time.time()
            valid_cluster_indices = self.adaptive_clustering.clustering(cloud_o3d.select_by_index(inlier_indices), trk)
            clustering_end = time.time()

            mean_clustering_time += clustering_end - clustering_start

            if valid_cluster_indices is None:
                continue

            clusters_vec[inlier_indices[valid_cluster_indices]] = (i+1)*100 + 1

            xx = cloud_np[inlier_indices[valid_cluster_indices]][:, 0]
            yy = cloud_np[inlier_indices[valid_cluster_indices]][:, 1]

            fitting_start = time.time()
            result = self.l_shape_fitting.fitting(xx, yy, trk)
            fitting_end = time.time()

            mean_fitting_time += fitting_end - fitting_start

            for est_bbox in result[0]:

                if est_bbox is None:
                    continue

                iou = boxoverlap(trk, est_bbox)
                if iou > self.iou_threshold:

                    zz = cloud_np[inlier_indices[valid_cluster_indices]][:, 2]
                    est_bbox.header = cloud_msg.header
                    est_bbox.dimensions.z = max(2.0, np.max(zz) - np.min(zz))
                    est_bbox.pose.position.z =  trk.pose.position.z + est_bbox.dimensions.z / 2
                    est_bbox.label = trk.label
                    est_bbox.value = 0.3
                    fine_tuned_tracks.boxes.append(est_bbox)


            # fine_tuned_box = self.fine_tune_cluster(cloud_np[inlier_indices], trk)
            #
            # if fine_tuned_box is not None:
            #     fine_tuned_box.header = cloud_msg.header
            #     fine_tuned_tracks.boxes.append(fine_tuned_box)

        print("=========================================")
        print(f"Callback Time: {time.time() - start_callback}")
        print(f"Total FineTune Time: {mean_clustering_time + mean_fitting_time}")
        print(f"Total Clustering Time: {mean_clustering_time}")
        print(f"Total Fitting Time: {mean_fitting_time}")

        mean_clustering_time /= len(extended_tracks.boxes)
        mean_fitting_time /= len(extended_tracks.boxes)

        print(f"Mean Clustering Time: {mean_clustering_time}")
        print(f"Mean Fitting Time: {mean_fitting_time}")
        print("=========================================")
        self.finetuned_tracks_pub.publish(fine_tuned_tracks)

        out_cloud = np.zeros(len(cloud_np),
                             dtype=[('x', np.float32),
                                    ('y', np.float32),
                                    ('z', np.float32),
                                    ('cropped', np.float32),
                                    ('clusters', np.float32)])

        out_cloud['x'] = cloud_np[:, 0]
        out_cloud['y'] = cloud_np[:, 1]
        out_cloud['z'] = cloud_np[:, 2]
        out_cloud['cropped'] = intensity_vec.reshape(-1)
        out_cloud['clusters'] = clusters_vec.reshape(-1)

        out_cloud_msg = rnp.msgify(PointCloud2, out_cloud)
        out_cloud_msg.header = cloud_msg.header
        self.debug_cloud_pub.publish(out_cloud_msg)



if __name__ == '__main__':
    rospy.init_node('tracker_guided_detector')
    tgd = TrackerGuidedDetector('/carla/ego_vehicle/lidar/measurements/tracked',
                                    '/carla/ego_vehicle/lidar/nonground',
                                    '/carla/ego_vehicle/lidar/tracks/fine_tuned')
    rospy.spin()