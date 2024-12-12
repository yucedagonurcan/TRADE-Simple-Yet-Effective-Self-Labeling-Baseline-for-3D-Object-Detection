import copy

import rosbag as rbag
import numpy as np

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from random import randint
import pickle
import os
from copy import deepcopy
from derived_object_msgs.msg import ObjectArray
from typing import List
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
import ros_numpy as rnp
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

TRIAL_BAGS=[
           "/home/yucedagonurcan/Documents/CARLA_Dataset_Report_Baseline_v1_2024-12-09-10-03-14.bag",
            "/home/yucedagonurcan/Documents/CARLA_Dataset_Report_v2_2024-12-11-21-09-50.bag",
]


PICKLE_PREFIX="trial_"

TRACKED_OBJECTS_TOPIC_BASE = "/carla/ego_vehicle/lidar/objects/tracked"
TRACKED_OBJECTS_TOPIC_FINE = "/carla/ego_vehicle/lidar/measurements/tracked"
GROUND_TRUTH_OBJECTS_TOPIC = "/carla/ego_vehicle/objects"
SEMANTIC_LIDAR_TOPIC = "/carla/ego_vehicle/lidar"

MIN_POINTS_FOR_VISIBILITY = 5
ASSOCIATION_MAX_COST = 0.8

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

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




def derived2bbox(derived_object_msg):
    bounding_box_msg = BoundingBoxArray()
    bounding_box_msg.header = derived_object_msg.header

    for obj in derived_object_msg.objects:
        bounding_box = BoundingBox()
        bounding_box.header = derived_object_msg.header
        bounding_box.pose = obj.pose
        bounding_box.dimensions.x = obj.shape.dimensions[0]
        bounding_box.dimensions.y = obj.shape.dimensions[1]
        bounding_box.dimensions.z = obj.shape.dimensions[2]
        bounding_box.label = obj.classification
        bounding_box.value = obj.id
        bounding_box_msg.boxes.append(bounding_box)

    return bounding_box_msg


def value_array_to_bounding_box(value_array):
    bounding_box = BoundingBox()
    bounding_box.pose.position.x = value_array[0]
    bounding_box.pose.position.y = value_array[1]
    bounding_box.pose.position.z = value_array[2]
    bounding_box.pose.orientation.x = value_array[3]
    bounding_box.pose.orientation.y = value_array[4]
    bounding_box.pose.orientation.z = value_array[5]
    bounding_box.pose.orientation.w = value_array[6]
    bounding_box.dimensions.x = value_array[7]
    bounding_box.dimensions.y = value_array[8]
    bounding_box.dimensions.z = value_array[9]
    bounding_box.label = value_array[10]
    bounding_box.value = value_array[11]

    return bounding_box


class Sequence:
    def __init__(self, timestamp):
        self.seq_id = randint(0, 1000)

        self.timestamp = timestamp.to_sec()
        self.pointcloud = None
        self.predictions = []
        self.ground_truth = []

        self.visible_object_ids = []

    def addPrediction(self, obj: BoundingBoxArray):
        for i, pred in enumerate(obj.boxes):
            self.predictions.append([pred.pose.position.x, pred.pose.position.y, pred.pose.position.z,
                                     pred.pose.orientation.x, pred.pose.orientation.y, pred.pose.orientation.z, pred.pose.orientation.w,
                                     pred.dimensions.x, pred.dimensions.y, pred.dimensions.z,
                                     pred.label, pred.value])

    def addGroundTruth(self, obj):

        gt_bbox = derived2bbox(obj)
        for i, gt in enumerate(gt_bbox.boxes):
            self.ground_truth.append([gt.pose.position.x, gt.pose.position.y, gt.pose.position.z,
                                      gt.pose.orientation.x, gt.pose.orientation.y, gt.pose.orientation.z, gt.pose.orientation.w,
                                      gt.dimensions.x, gt.dimensions.y, gt.dimensions.z,
                                      gt.label, gt.value])
    def setPointcloud(self, pointcloud: PointCloud2):
        self.pointcloud = pointcloud

    def setVisibleObjectIds(self, visible_object_ids):
        self.visible_object_ids = visible_object_ids

    def getVisibleGroundTruth(self):
        return [gt for gt in self.ground_truth if gt[11] in self.visible_object_ids]

MAX_DIST_ASSOCIATION = 10.0
def calculate_cost(gt: BoundingBox, pred: BoundingBox):
    # Calculate the cost of associating the ground truth and the prediction
    # The cost is the distance between the centers of the two bounding boxes
    # If the distance is greater than a threshold, return a high cost
    # If the distance is less than a threshold, return the distance

    gt_center = np.array([gt.pose.position.x, gt.pose.position.y, gt.pose.position.z])
    pred_center = np.array([pred.pose.position.x, pred.pose.position.y, pred.pose.position.z])

    dist = np.linalg.norm(gt_center - pred_center)

    if dist > MAX_DIST_ASSOCIATION:
        return 1.0
    else:
        return 1.0-boxoverlap(gt, pred)


def calculate_cost_affinity(sequence: Sequence)->List[int]:

    visible_gt = sequence.getVisibleGroundTruth()
    cost_affinity = np.zeros((len(visible_gt), len(sequence.predictions)))

    for i, gt in enumerate(visible_gt):
        gt_bbox = value_array_to_bounding_box(gt)

        for j, pred in enumerate(sequence.predictions):
            cost_affinity[i, j] = calculate_cost(gt_bbox, value_array_to_bounding_box(pred))

    return cost_affinity


def evaluate(sequences: List[Sequence]):

    # Evaluate the sequences
    # FalsePositives, FalseNegatives, Number of Ground Truths
    metrics = np.zeros((len(sequences), 3))

    for i, seq in enumerate(tqdm(sequences)):


        # Associate the predictions with the ground truth
        cost_affinity = calculate_cost_affinity(seq)

        row_ind, col_ind = linear_sum_assignment(cost_affinity)
        matched_indices = np.stack((row_ind, col_ind), axis=1)

        filtered_matches = []
        for m in matched_indices:
            if cost_affinity[m[0], m[1]] < ASSOCIATION_MAX_COST:
                filtered_matches.append(m)

        false_pos = len(seq.predictions) - len(filtered_matches)
        false_neg = len(seq.visible_object_ids) - len(filtered_matches)
        num_gt = len(seq.visible_object_ids)

        metrics[i, 0] = false_pos
        metrics[i, 1] = false_neg
        metrics[i, 2] = num_gt


    mota = 1 - np.sum(metrics[:, 0] + metrics[:, 1]) / np.sum(metrics[:, 2])
    print(f"MOTA: {mota}")
    return mota, metrics


if __name__ == "__main__":

    for i, bag in enumerate(TRIAL_BAGS):
        sequences = []
        pickle_name = PICKLE_PREFIX + str(i) + ".pkl"

        if os.path.exists(pickle_name):
            print("Pickle file exists, loading...")
            with open(pickle_name, "rb") as f:
                sequences = pickle.load(f)
        else:

            print("Loading bag: ", bag)
            ground_truth_objects = []
            predicted_objects = []
            pointcloud_array = []


            bag = rbag.Bag(bag)
            for topic, msg, t in bag.read_messages(topics=[TRACKED_OBJECTS_TOPIC_BASE, TRACKED_OBJECTS_TOPIC_FINE, GROUND_TRUTH_OBJECTS_TOPIC, SEMANTIC_LIDAR_TOPIC]):
                if topic == TRACKED_OBJECTS_TOPIC_BASE:
                    predicted_objects.append(msg)
                elif topic == TRACKED_OBJECTS_TOPIC_FINE:
                    predicted_objects.append(msg)

                if topic == GROUND_TRUTH_OBJECTS_TOPIC:
                    ground_truth_objects.append(msg)

                if topic == SEMANTIC_LIDAR_TOPIC:
                    pointcloud_array.append(msg)

            bag.close()
            print(f"Loaded {len(predicted_objects)} predictions and {len(ground_truth_objects)} ground truth objects")

            # with open(pickle_name, "wb") as f:
            #     pickle.dump([predicted_objects, ground_truth_objects], f)

            sequences = []
            # Create sequences
            for i, gt_obj in enumerate(ground_truth_objects):
                seq = Sequence(gt_obj.header.stamp)

                for j, pred_obj in enumerate(predicted_objects):
                    if pred_obj.header.stamp < gt_obj.header.stamp:
                        continue
                    if pred_obj.header.stamp > gt_obj.header.stamp:
                        break

                    for k, pointcloud in enumerate(pointcloud_array):
                        if pointcloud.header.stamp < gt_obj.header.stamp:
                            continue
                        if pointcloud.header.stamp > gt_obj.header.stamp:
                            break

                        seq.addGroundTruth(gt_obj)
                        seq.addPrediction(predicted_objects[i])

                        pc_array = rnp.point_cloud2.pointcloud2_to_array(pointcloud)
                        seq.setPointcloud(pc_array)

                        # Get all object ids that are visible in the pointcloud
                        mentioned_object_ids = np.unique(pc_array['ObjIdx'])

                        visible_object_ids = []

                        # If object has at least 5 points, add it to the visible object ids
                        for obj_id in mentioned_object_ids:
                            if len(np.where(pc_array['ObjIdx'] == obj_id)[0]) > MIN_POINTS_FOR_VISIBILITY:
                                visible_object_ids.append(obj_id)


                        seq.setVisibleObjectIds(visible_object_ids)
                        sequences.append(seq)


            print(f"Created {len(sequences)} sequences")
            with open(pickle_name, "wb") as f:
                pickle.dump(sequences, f)

            print("Saved sequences to sequences.pkl")

        mota_score = 0
        metrics = None

        if os.path.exists(f"metrics_{i}.pkl"):
            print(f"Metrics file for trial {i} exists, skipping evaluation")

            with open(f"metrics_{i}.pkl", "rb") as f:
                metrics = pickle.load(f)
                mota_score = 1 - np.sum(metrics[:, 0] + metrics[:, 1]) / np.sum(metrics[:, 2])
        else:
            print("Evaluating sequences...")
            # Evaluate sequences
            mota_score, metrics = evaluate(sequences)

            # Save the metrics
            with open(f"metrics_{i}.csv", "wb") as f:
                import pandas as pd
                pd.DataFrame(metrics).to_csv(f)

        print(f"MOTA score for trial {i}: {mota_score}")
