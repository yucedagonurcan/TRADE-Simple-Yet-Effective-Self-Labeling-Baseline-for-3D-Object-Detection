from derived_object_msgs.msg import ObjectArray
import rospy
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

class DerivedObjectToBoundingBox:
    def __init__(self, derived_object_topic, bounding_box_topic):
        self.derived_object_topic = derived_object_topic
        self.bounding_box_topic = bounding_box_topic

        self.bounding_box_pub = rospy.Publisher(bounding_box_topic, BoundingBoxArray, queue_size=1)
        self.derived_object_sub = rospy.Subscriber(derived_object_topic, ObjectArray, self.derived_object_callback)

    def derived_object_callback(self, derived_object_msg):
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

        self.bounding_box_pub.publish(bounding_box_msg)

if __name__ == "__main__":
    rospy.init_node("derived_object_to_bounding_box")
    derived_object_topic = rospy.get_param("~derived_object_topic", "/carla/ego_vehicle/objects")
    bounding_box_topic = rospy.get_param("~bounding_box_topic", "/ground_truth_objects")
    derived_object_to_bounding_box = DerivedObjectToBoundingBox(derived_object_topic, bounding_box_topic)
    rospy.spin()

