import carla
import rospy
import ros_numpy as rnp
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
from enum import Enum

class LiDARConfig:
    width = 1024
    height = 64

class ObjectTag(Enum):
    Roads = 1, 
    Sidewalks = 2, 
    Buildings = 3,
    Walls = 4, 
    Fences = 5, 
    Poles = 6, 
    TrafficLight = 7,
    TrafficSigns = 8, 
    Vegetation = 9, 
    Terrain = 10, 
    Sky = 11,
    Pedestrians = 12, 
    Rider = 13, 
    Car = 14, 
    Truck = 15,
    Bus = 16, 
    Train = 17, 
    Motorcycle = 18, 
    Bicycle = 19,
    Static = 20, 
    Dynamic = 21, 
    Other = 22, 
    Water = 23,
    RoadLines = 24, 
    Ground = 25, 
    Bridge = 26, 
    RailTrack = 27,
    GuardRail = 28,

def normalize_and_scale_vector(vector, scale):
    vector_norm = (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector_norm * scale

class LiDARFieldImageGenerator:
    def __init__(self, port=2000):
        self.sub_cloud = rospy.Subscriber('/carla/ego_vehicle/lidar', PointCloud2, self.callback, queue_size=1)
        
        self.pub_cloud = rospy.Publisher('/carla/ego_vehicle/lidar_intensity', PointCloud2, queue_size=1)
        self.pub_lidar_img = rospy.Publisher('/carla/ego_vehicle/lidar_intensity_img', Image, queue_size=1)
        
        
        self.attenuation_rate=0.004
        
        
        self.metalic_objects = [
            ObjectTag.Car,
            ObjectTag.Bicycle,
            ObjectTag.Motorcycle,
            ObjectTag.Truck,
            ObjectTag.Bus
        ]
        
        self.human_objects = [
            ObjectTag.Pedestrians,
            ObjectTag.Rider
        ]
        
        self.retroreflective_objects = [
            ObjectTag.RoadLines,
            ObjectTag.GuardRail,
            ObjectTag.TrafficSigns,
            ObjectTag.TrafficLight
        ]
        
        self.vegetation_objects = [
            ObjectTag.Vegetation
        ]
        
        self.building_objects = [
            ObjectTag.Buildings
        ]

        self.classes_of_interest =  self.metalic_objects +\
                                    self.human_objects +\
                                    self.retroreflective_objects +\
                                    self.vegetation_objects +\
                                    self.building_objects
        
        # Center, deviation
        self.metalic_reflectivity = (150, 40)
        self.retroreflective_reflectivity = (220, 30)
        self.human_reflectivity = (80, 40)
        self.vegetation_reflectivity = (10, 10)
        self.building_reflectivity = (20, 10)
        self.any_other_reflectivity = (10, 5)
        
        self.cos_angle_percentage = 0.7
        self.cos_angle_scale = 2
        
        self.ignore_classes = [ i for i in ObjectTag if i not in self.classes_of_interest]
        
        self.ground_height_deviation = 0.2
        self.gradient_threshold = 0.5
        
    

    def callback(self, data):
        sensor_data = rnp.point_cloud2.pointcloud2_to_array(data)
        
        material_scaling = self.calculate_material_scalings(sensor_data)
        distance_loss = self.calculate_distance_loss(sensor_data)
        material_loss_wrt_distance = distance_loss * material_scaling 
        surface_angle_loss = self.calculate_cos_angle_loss(sensor_data, material_loss_wrt_distance)

        intensity = material_loss_wrt_distance - surface_angle_loss

        final_intensity = normalize_and_scale_vector(intensity, 255)
        
        sensor_data_np = np.array(sensor_data.tolist(), dtype=np.float32)
        sensor_data_np = np.append(sensor_data_np[:, :3], final_intensity.reshape(-1, 1), axis=1)
        
        cloud_arr_az_el = self.unorganized_xyzi_to_organized_xyziaed(sensor_data_np)
        
        dist_grad = self.calculate_dist_gradient(cloud_arr_az_el)
        dist_grad_norm = normalize_and_scale_vector(dist_grad, 255)


        # Set the ground points' gradient to 0
        dist_grad_norm[cloud_arr_az_el[:,:,2] < self.ground_height_deviation - 2.3] = 0
        
        
        # Create a image from the intensity data
        img = np.zeros((64, 1024, 3), dtype=np.uint8)
        img[:,:,0] = cloud_arr_az_el[:,:,3]
        img[:,:,1] = cloud_arr_az_el[:,:,3]
        img[:,:,2] = dist_grad_norm
        
        img_msg = rnp.image.numpy_to_image(img, 'rgb8')
        img_msg.header.stamp = data.header.stamp
        img_msg.header.frame_id = data.header.frame_id
        self.pub_lidar_img.publish(img_msg)
        
        
        final_cloud = np.zeros((LiDARConfig.height, LiDARConfig.width),
                               dtype=[('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('intensity', np.float32),
                                        ('azimuth', np.float32),
                                        ('elevation', np.float32),
                                        ('distance', np.float32)])
        
        final_cloud['x'] = cloud_arr_az_el[:,:,0]
        final_cloud['y'] = cloud_arr_az_el[:,:,1]
        final_cloud['z'] = cloud_arr_az_el[:,:,2]
        final_cloud['intensity'] = cloud_arr_az_el[:,:,3]
        final_cloud['azimuth'] = cloud_arr_az_el[:,:,4]
        final_cloud['elevation'] = cloud_arr_az_el[:,:,5]
        final_cloud['distance'] = cloud_arr_az_el[:,:,6]
        
        cloud_msg = rnp.point_cloud2.array_to_pointcloud2(final_cloud, data.header.stamp, data.header.frame_id)
        self.pub_cloud.publish(cloud_msg)
        
        print("Received LiDAR intensity data")
    
    def calculate_dist_gradient(self, cloud):
        dist_grad = np.gradient(cloud[:,:,6], axis=1)
        dist_grad = np.abs(dist_grad)
        
        dist_grad[dist_grad > self.gradient_threshold] = 1
        dist_grad[dist_grad < self.gradient_threshold] = 0
        
        return dist_grad
    
    def calculate_cos_angle_loss(self, raw_cloud, activation):
        return (raw_cloud['CosAngle']**2) * activation * self.cos_angle_percentage 
    
    def calculate_distance_loss(self, raw_cloud):
        dist = np.sqrt(raw_cloud['x']**2 + raw_cloud['y']**2 + raw_cloud['z']**2)
        dist = np.clip(dist, 1, 100)
        distance_loss = np.exp(-self.attenuation_rate * dist)
        return distance_loss

    def calculate_material_scalings(self, raw_cloud):

        material_scalings = np.zeros(raw_cloud.shape[0])
                
        for ignore_class in self.ignore_classes:
            
            num_points = np.sum(raw_cloud["ObjTag"] == ignore_class.value)
            distribution = np.random.normal(self.any_other_reflectivity[0], self.any_other_reflectivity[1], num_points)
            material_scalings[raw_cloud["ObjTag"] == ignore_class.value] = distribution
            
        for metalic_class in self.metalic_objects:
            num_points = np.sum(raw_cloud["ObjTag"] == metalic_class.value)
            distribution = np.random.normal(self.metalic_reflectivity[0], self.metalic_reflectivity[1], num_points)
            material_scalings[raw_cloud["ObjTag"] == metalic_class.value] = distribution
            
        for human_class in self.human_objects:
            num_points = np.sum(raw_cloud["ObjTag"] == human_class.value)
            distribution = np.random.normal(self.human_reflectivity[0], self.human_reflectivity[1], num_points)
            material_scalings[raw_cloud["ObjTag"] == human_class.value] = distribution
            
        for retroreflective_class in self.retroreflective_objects:
            num_points = np.sum(raw_cloud["ObjTag"] == retroreflective_class.value)
            distribution = np.random.normal(self.retroreflective_reflectivity[0], self.retroreflective_reflectivity[1], num_points)
            material_scalings[raw_cloud["ObjTag"] == retroreflective_class.value] = distribution
            
        for vegetation_class in self.vegetation_objects:
            num_points = np.sum(raw_cloud["ObjTag"] == vegetation_class.value)
            distribution = np.random.normal(self.vegetation_reflectivity[0], self.vegetation_reflectivity[1], num_points)
            material_scalings[raw_cloud["ObjTag"] == vegetation_class.value] = distribution
            
        for building_class in self.building_objects:
            num_points = np.sum(raw_cloud["ObjTag"] == building_class.value)
            distribution = np.random.normal(self.building_reflectivity[0], self.building_reflectivity[1], num_points)
            material_scalings[raw_cloud["ObjTag"] == building_class.value] = distribution
            
            
        material_scalings = (material_scalings - np.min(material_scalings)) / (np.max(material_scalings) - np.min(material_scalings))
        
        return material_scalings

    def unorganized_xyzi_to_organized_xyziaed(self, cloud):

        # For each point, calculate azimuth and elevation
        azimuth = np.arctan2(cloud[:, 1], cloud[:, 0])
        dist = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2 + cloud[:, 2]**2)
        elevation = np.arctan2(cloud[:, 2], dist)
        
        
        min_azimuth = np.min(azimuth)
        max_azimuth = np.max(azimuth)
        
        min_elevation = np.min(elevation)
        max_elevation = np.max(elevation)
        
        azimuth_indices     = (LiDARConfig.width - (((azimuth - min_azimuth) / (max_azimuth - min_azimuth)) * LiDARConfig.width)).astype(int)
        elevation_indices   = (LiDARConfig.height - (((elevation - min_elevation) / (max_elevation - min_elevation)) * LiDARConfig.height)).astype(int)
        
        
        azimuth_indices = np.clip(azimuth_indices, 0, LiDARConfig.width-1)
        elevation_indices = np.clip(elevation_indices, 0, LiDARConfig.height-1)
        
        
        organized_cloud = np.zeros((LiDARConfig.height, LiDARConfig.width, 7), dtype=np.float32)     
        
        el_az_index = np.stack((elevation_indices, azimuth_indices), axis=-1)
        
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 0] = cloud[:, 0]
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 1] = cloud[:, 1]
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 2] = cloud[:, 2]
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 3] = cloud[:, 3]
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 4] = azimuth
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 5] = elevation
        organized_cloud[el_az_index[:,0], el_az_index[:,1], 6] = dist
        
        
        return organized_cloud

        
if __name__ == '__main__':
    rospy.init_node('carla_lidar_intensity', anonymous=False)
    lidar_intensity = LiDARFieldImageGenerator()
    rospy.spin()