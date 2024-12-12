#!/home/yucedagonurcan/miniconda3/envs/openmmlab/bin python3
import numpy as np

from typing import List
from mmdet3d.apis import LidarDet3DInferencer

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import quaternion_from_euler, quaternion_matrix
from geometry_msgs.msg import Point

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

import copy

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from enum import Enum

import os

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

color_map = {
    0: (0, 1, 0, 1),
    1: (1, 0, 0, 1),
    2: (0, 0, 1, 1),
    3: (1, 1, 0, 1),
    4: (1, 0, 1, 1),
    5: (0, 1, 1, 1),
    6: (1, 0.5, 0, 1),
    7: (0, 1, 0.5, 1),
    8: (0.5, 0, 1, 1),
    9: (1, 0.5, 0.5, 1)
}

class Classification(Enum):
    PEDESTRIAN = 0
    CAR = 2
    CYCLIST = 1

def get_transform(target_frame: str, source_frame: str, time: rospy.Time, timeout: rospy.Duration, buffer: Buffer) -> TransformStamped:
    try:
        transform = buffer.lookup_transform(target_frame, source_frame, time, timeout)
    except Exception as e:
        return None
    return transform

def create_transform_matrix(transform: TransformStamped) -> np.ndarray:

    translation = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    rotation = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])

    matrix = quaternion_matrix(rotation)
    matrix[:3, 3] = translation

    return matrix

# Transform each point in the list of points to the first pointcloud's frame
def collate_points(points: List[np.ndarray], transforms: List[np.ndarray], timestamps: List[rospy.Time]) -> List[np.ndarray]:

    first_transform = transforms[0]
    collated_points = copy.deepcopy(points)

    sweep_ts = timestamps[0]

    for i in range(len(points)):

        pc2map = transforms[i]
        pc2sensor = np.linalg.inv(first_transform) @ pc2map

        collated_points[i][:, :3] = (pc2sensor[:3, :3] @ points[i][:, :3].T + pc2sensor[:3, 3].reshape(-1, 1)).T
        collated_points[i][:, -1] = timestamps[i].to_sec() - sweep_ts.to_sec()

    return collated_points

def remove_close(   points: np.ndarray,
                    radius: float = 1.0) -> np.ndarray:
    """Remove point too close within a certain radius from origin.

    Args:
        points (np.ndarray | :obj:`BasePoints`): Sweep points.
        radius (float): Radius below which points are removed.
            Defaults to 1.0.

    Returns:
        np.ndarray | :obj:`BasePoints`: Points after removing.
    """

    points_numpy = points

    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]


def convert_organized_pc(pointcloud: PointCloud2):
    cur_points = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud)
    cur_points = cur_points.flatten()

    return cur_points

def convert_unorganized_pc(pointcloud: PointCloud2):
    cur_points = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud)
    cur_points = cur_points.flatten()

    return cur_points

def normalize_to_range(points: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Normalize points to a certain range.

    Args:
        points (np.ndarray): Points to be normalized.
        min_val (float): Minimum value.
        max_val (float): Maximum value.

    Returns:
        np.ndarray: Normalized points.
    """
    assert min_val < max_val
    min_p = points.min()
    max_p = points.max()
    points = (points - min_p) / (max_p - min_p) * (max_val - min_val) + min_val
    return points

def shuffle_points(points: np.ndarray) -> np.ndarray:
    """Shuffle points.

    Args:
        points (np.ndarray): Points to be shuffled.

    Returns:
        np.ndarray: Shuffled points.
    """
    np.random.shuffle(points)
    return points

class CenterPointROS():

    def __init__(self, model_config_file: str, model_checkpoint_file: str, topic_name: str, is_organized: bool = False):

        self.lidar_inferencer = LidarDet3DInferencer(   model=model_config_file,
                                                        weights=model_checkpoint_file,
                                                        device='cuda:0',
                                                        show_progress=False)
        self.pc_sub = rospy.Subscriber(topic_name, PointCloud2, self.infer, queue_size=100)
        self.marker_pub = rospy.Publisher('boxes', MarkerArray, queue_size=1, latch=True)


        self.objects_pub = rospy.Publisher(topic_name+'/objects', BoundingBoxArray, queue_size=1, latch=True)
        self.densified_pc_pub = rospy.Publisher(topic_name+'/densified_pc', PointCloud2, queue_size=1, latch=True)

        self.sweep_points = []
        self.sweep_transforms = []
        self.sweep_ts = []

        self.num_sweeps = 4

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.message_fields = ["x", "y", "z", "timestamp"]
        self.map_frame = "map"

        self.score_threshold = 0.25

        self.classes_to_show = [0, # car
                                1, # truck
                                2, # construction_vehicle
                                3, # bus
                                4, # trailer
                                6, # motorcycle
                                7, # bicycle
                                8] # pedestrian

        self.class_mapping = {  0: Classification.CAR.value,
                                1: Classification.CAR.value,
                                2: Classification.CAR.value,
                                3: Classification.CAR.value,
                                4: Classification.CAR.value,
                                6: Classification.CYCLIST.value,
                                7: Classification.CYCLIST.value,
                                8: Classification.PEDESTRIAN.value}

        self.conversion_func = convert_organized_pc if is_organized else convert_unorganized_pc


    def infer(self, pointcloud: PointCloud2):
        # cur_points = pointcloud.astype(dtype=np.float32).reshape(-1, 4)
        cur_points = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud)
        cur_points = cur_points.flatten()

        # Add dummy intensity values
        points_with_ts = np.zeros((cur_points.shape[0], len(self.message_fields)),dtype=np.float32)

        points_with_ts[:, 0] = cur_points['x']
        points_with_ts[:, 1] = cur_points['y']
        points_with_ts[:, 2] = cur_points['z']


        if len(self.sweep_points) == self.num_sweeps:
            self.sweep_points.pop(0)
            self.sweep_transforms.pop(0)
            self.sweep_ts.pop(0)

        # Add timestamps
        cur_transform = get_transform(self.map_frame,
                                      pointcloud.header.frame_id,
                                      pointcloud.header.stamp,
                                      rospy.Duration(0.1),
                                      self.tf_buffer)
        if cur_transform is None:
            return

        cur_transform_matrix = create_transform_matrix(cur_transform)

        self.sweep_points.append(points_with_ts)
        self.sweep_transforms.append(cur_transform_matrix)
        self.sweep_ts.append(pointcloud.header.stamp)

        if len(self.sweep_points) < self.num_sweeps:
            return

        input_dict = {}

        collated_pts = collate_points(self.sweep_points, self.sweep_transforms, self.sweep_ts)

        densified_pc = np.concatenate(collated_pts, axis=0)

        if self.densified_pc_pub.get_num_connections() > 0:
            densified_arr = np.zeros(densified_pc.shape[0],
                                     dtype=[
                                         (field, np.float32) for field in self.message_fields])

            densified_arr["x"] = densified_pc[:, 0]
            densified_arr["y"] = densified_pc[:, 1]
            densified_arr["z"] = densified_pc[:, 2]
            densified_arr["timestamp"] = densified_pc[:, 3]

            densified_pc_msg = ros_numpy.msgify(PointCloud2, densified_arr, frame_id=self.map_frame)
            densified_pc_msg.header = pointcloud.header

            self.densified_pc_pub.publish(densified_pc_msg)


        input_dict["points"] = densified_pc

        # Run inference
        result = self.lidar_inferencer(inputs=input_dict,
                                       show=False,
                                       out_dir="",
                                       wait_time=0.1,
                                       pred_score_thr=0.7,
                                       no_save_pred=True)
        self.lidar_inferencer.num_visualized_frames += 1

        bboxes = result["predictions"][0]["bboxes_3d"]
        scores = result["predictions"][0]["scores_3d"]
        labels = result["predictions"][0]["labels_3d"]

        out_objects = BoundingBoxArray()
        out_objects.header.frame_id = pointcloud.header.frame_id
        out_objects.header.stamp = self.sweep_ts[0]

        for i in range(len(bboxes)):

            bbox = bboxes[i]
            score = scores[i]
            label = labels[i]

            if label not in self.classes_to_show:
                continue
            if score < self.score_threshold:
                continue

            current_object = BoundingBox()

            current_object.header.frame_id = pointcloud.header.frame_id
            current_object.header.stamp = self.sweep_ts[0]

            current_object.pose.position.x = bbox[0]
            current_object.pose.position.y = bbox[1]
            current_object.pose.position.z = bbox[2] + bbox[5] / 2

            q = quaternion_from_euler(0, 0, bbox[6])
            current_object.pose.orientation.x = q[0]
            current_object.pose.orientation.y = q[1]
            current_object.pose.orientation.z = q[2]
            current_object.pose.orientation.w = q[3]

            current_object.dimensions.x = bbox[3]
            current_object.dimensions.y = bbox[4]
            current_object.dimensions.z = bbox[5]

            current_object.label = self.class_mapping[label]
            current_object.value = score

            out_objects.boxes.append(current_object)

        self.objects_pub.publish(out_objects)

def main():
    rospy.init_node('inference_module_centerpoint')

    config_file = f"{SCRIPT_PATH}/model_data/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
    checkpoint_file = f"{SCRIPT_PATH}/model_data/epoch_30.pth"

    CenterPointROS(config_file, checkpoint_file, '/carla/ego_vehicle/lidar')
    rospy.spin()



if __name__ == "__main__":
    main()