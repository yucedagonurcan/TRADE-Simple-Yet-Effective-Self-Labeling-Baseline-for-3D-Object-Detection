import rospy
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading

# Define a lock
lock = threading.Lock()

MAX_DIST_ASSOCIATION = 0.5
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



class ObjectListFusion:
    def __init__(self, detections_topic, fine_tuned_topic, output_topic):
        self.detections_topic = detections_topic
        self.fine_tuned_topic = fine_tuned_topic
        self.output_topic = output_topic

        self.fused_objects_pub = rospy.Publisher(output_topic, BoundingBoxArray, queue_size=1, latch=True)

        self.detections_sub = rospy.Subscriber(detections_topic, BoundingBoxArray, self.detections_callback, queue_size=10)
        self.fine_tuned_sub = rospy.Subscriber(fine_tuned_topic, BoundingBoxArray, self.fine_tuned_callback, queue_size=10)


        self.fine_tuned_history = []
        self.detection_history = []

        self.max_age = rospy.Duration(0.2)

    def fine_tuned_callback(self, fine_tuned_msg):

        # Lock the thread
        with lock:
            # Append the fine tuned message to the history

            self.fine_tuned_history.append(fine_tuned_msg)

    def detections_callback(self, detections_msg):

        # Lock the thread
        with lock:
            print("--------------------")
            print(f"Incoming Detection {detections_msg.header.stamp.to_sec()}")
            print("Current State of the Detection History")

            for i, det in enumerate(self.detection_history):
                print(f"\t - Detection {i}: {det.header.stamp.to_sec()}")

            print("Current State of the Fine Tuned History")
            for j, fine_tuned in enumerate(self.fine_tuned_history):
                print(f"\t - Fine Tuned {j}: {fine_tuned.header.stamp.to_sec()}")

            if len(self.fine_tuned_history) == 0:
                self.fused_objects_pub.publish(detections_msg)
                print("No fine tuned objects found")
                print(f"Published the detection message {detections_msg.header.stamp.to_sec()}")
                return

            self.detection_history.append(detections_msg)

            related_fine_tuned_idx = None

            for i, cur_fine_tuned in enumerate(self.fine_tuned_history):
                for j, cur_detection in enumerate(self.detection_history):

                    if abs(cur_fine_tuned.header.stamp.to_sec() - cur_detection.header.stamp.to_sec()) < 10e-3:
                        related_fine_tuned_idx = i, j
                        break


            if related_fine_tuned_idx is not None:

                self.fuse(self.detection_history[related_fine_tuned_idx[1]],
                          self.fine_tuned_history[related_fine_tuned_idx[0]])
                print(f"Published the fused object message {self.detection_history[related_fine_tuned_idx[1]].header.stamp.to_sec()}")


                self.fine_tuned_history = self.fine_tuned_history[related_fine_tuned_idx[0]+1:]
                self.detection_history = self.detection_history[related_fine_tuned_idx[1]+1:]
            else:
                self.fused_objects_pub.publish(detections_msg)
                print("No related fine tuned object found")
                print(f"Published the detection message {detections_msg.header.stamp.to_sec()}")
                self.detection_history.pop(-1)

    def fuse(self, detections_msg, fine_tuned_msg):
        fused_objects = BoundingBoxArray()
        fused_objects.header = detections_msg.header


        associations = []

        for i, det_obj in enumerate(detections_msg.boxes):
                det_obj_center = np.array([det_obj.pose.position.x, det_obj.pose.position.y])
                max_edge_det = max(det_obj.dimensions.x, det_obj.dimensions.y)

                for j, fine_tuned_obj in enumerate(fine_tuned_msg.boxes):

                    fine_tuned_obj_center = np.array([fine_tuned_obj.pose.position.x, fine_tuned_obj.pose.position.y])
                    max_edge_fine_tuned = max(fine_tuned_obj.dimensions.x, fine_tuned_obj.dimensions.y)

                    dist = np.linalg.norm(det_obj_center - fine_tuned_obj_center)

                    if dist - max_edge_det - max_edge_fine_tuned < MAX_DIST_ASSOCIATION:
                        iou = boxoverlap(det_obj, fine_tuned_obj)
                        if iou > 0.3:
                            associations.append((i, j))

        for i, det_obj in enumerate(detections_msg.boxes):
            if i not in [x[0] for x in associations]:
                fused_objects.boxes.append(det_obj)

        for j, fine_tuned_obj in enumerate(fine_tuned_msg.boxes):
            if j not in [x[1] for x in associations]:
                fused_objects.boxes.append(fine_tuned_obj)

        for k in associations:
            fine_tuned_obj = fine_tuned_msg.boxes[k[1]]
            fused_objects.boxes.append(fine_tuned_obj)

        self.fused_objects_pub.publish(fused_objects)
        self.last_published_detection = detections_msg

def main():
    rospy.init_node('object_list_fusion')

    ObjectListFusion("/carla/ego_vehicle/lidar/objects",
                    '/carla/ego_vehicle/lidar/tracks/fine_tuned',
                    '/carla/ego_vehicle/lidar/measurements')
    rospy.spin()

if __name__ == '__main__':
    main()
