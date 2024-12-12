from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import rospy
from sensor_msgs.msg import PointCloud2

class Player:
    def __init__(self, in_cloud_topic, out_cloud_topic, condition_object_topic):
        self.in_cloud_topic = in_cloud_topic
        self.out_cloud_topic = out_cloud_topic
        self.condition_object_topic = condition_object_topic

        self.in_cloud_sub = rospy.Subscriber(in_cloud_topic, PointCloud2, self.in_cloud_callback, queue_size=100)
        self.out_cloud_pub = rospy.Publisher(out_cloud_topic, PointCloud2, queue_size=100, latch=True)

        self.condition_object_sub = rospy.Subscriber(condition_object_topic, BoundingBoxArray, self.condition_object_callback, queue_size=10)


        self.cloud_queue = []
        self.condition_object = None
        self.cloud = None

        self.time_margin = 0.05

    def in_cloud_callback(self, cloud_msg):
        self.cloud_queue.append(cloud_msg)

        if self.condition_object is not None:
            publish_cloud_until = self.condition_object.header.stamp.to_sec() + self.time_margin

            cloud_cursor = 0
            while len(self.cloud_queue) > 0:
                cloud = self.cloud_queue[cloud_cursor]

                if cloud.header.stamp.to_sec() < publish_cloud_until:
                    self.out_cloud_pub.publish(cloud)
                    cloud_cursor += 1
                else:
                    break

            self.cloud_queue = self.cloud_queue[cloud_cursor+1:]
        else:
            self.out_cloud_pub.publish(cloud_msg)

    def condition_object_callback(self, condition_object_msg):
        if len(self.cloud_queue) == 0:
            return
        self.condition_object = condition_object_msg



def main():
    rospy.init_node('conditioned_player')
    player = Player('/carla/ego_vehicle/lidar/tmp', '/carla/ego_vehicle/lidar', '/carla/ego_vehicle/lidar/tracks/fine_tuned')
    rospy.spin()

if __name__ == '__main__':
    main()
