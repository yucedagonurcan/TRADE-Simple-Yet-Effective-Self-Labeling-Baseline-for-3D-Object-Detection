#Object shape recognition with L-shape fitting

import numpy as np
import itertools
from enum import Enum
from jsk_recognition_msgs.msg import BoundingBox
from scipy.spatial.transform import Rotation
from enum import Enum
show_animation = True


class Classification(Enum):
    PEDESTRIAN = 0
    CAR = 2
    CYCLIST = 1

class ShapeRestrictions(Enum):
    CAR_MIN_LENGTH = 4.5
    CAR_MAX_LENGTH = 12.0

    CAR_MIN_WIDTH = 1.8
    CAR_MAX_WIDTH = 2.5

    PED_MAX_WIDTH = 2.0
    PED_MIN_WIDTH = 0.5

    PED_MIN_LENGTH = 0.5
    PED_MAX_LENGTH = 2.0

    CYCLIST_MIN_WIDTH = 0.5
    CYCLIST_MAX_WIDTH = 2.0

    CYCLIST_MIN_LENGTH = 0.5
    CYCLIST_MAX_LENGTH = 2.0


class LShapeFitting():

    class Criteria(Enum):
        AREA = 1
        CLOSENESS = 2
        VARIANCE = 3

    def __init__(self):
        # Parameters
        self.criteria = self.Criteria.CLOSENESS
        self.min_dist_of_closeness_crit = 0.1  # [m]
        self.dtheta_deg_for_serarch = 5.0  # [deg]
        self.R0 = 3.0  # [m] range segmentation param
        self.Rd = 0.1  # [m] range segmentation param
        self.range_margin = 10 # deg

    def fitting(self, ox, oy, template_object:BoundingBox=None):

        # Adaptive Range Segmentation
        # idsets = self._adoptive_range_segmentation(ox, oy)

        # Rectangle search
        rects = []
        # for ids in idsets:  # for each cluster
        cx = ox
        cy = oy
        rects.append(self._rectangle_search(cx, cy, template_object))

        return rects, None

    def _calc_area_criterion(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)

        alpha = -(c1_max - c1_min) * (c2_max - c2_min)

        return alpha

    def _calc_closeness_criterion(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)

        # D1 = [min([np.linalg.norm(c1_max - ic1),
        #            np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
        # D2 = [min([np.linalg.norm(c2_max - ic2),
        #            np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]

        D11 = np.min(np.abs(np.array([[c1_max - c1], [c1 - c1_min]])), axis=0)
        D22 = np.min(np.abs(np.array([[c2_max - c2], [c2 - c2_min]])), axis=0)
        beta = 0
        # for i, _ in enumerate(D1):
        #     d = max(min([D1[i], D2[i]]), self.min_dist_of_closeness_crit)
        #     beta += (1.0 / d)

        d = np.maximum(np.minimum(D11, D22), self.min_dist_of_closeness_crit)
        beta = np.sum(1.0 / d)
        return beta

    def _calc_variance_criterion(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)

        D1 = [min([np.linalg.norm(c1_max - ic1),
                   np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
        D2 = [min([np.linalg.norm(c2_max - ic2),
                   np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]

        E1, E2 = [], []
        for (d1, d2) in zip(D1, D2):
            if d1 < d2:
                E1.append(d1)
            else:
                E2.append(d2)

        V1 = 0.0
        if E1:
            V1 = - np.var(E1)

        V2 = 0.0
        if E2:
            V2 = - np.var(E2)

        gamma = V1 + V2

        return gamma

    def _rectangle_search(self, x, y, template_object:BoundingBox=None):

        X = np.array([x, y]).T

        dtheta = np.deg2rad(self.dtheta_deg_for_serarch)
        minp = (-float('inf'), None)


        starting_theta = 0.0
        end_theta = np.pi / 2.0

        if template_object is not None:
            quat = Rotation.from_quat([template_object.pose.orientation.x, template_object.pose.orientation.y, template_object.pose.orientation.z, template_object.pose.orientation.w])
            center_theta = quat.as_euler('xyz')[2]

            starting_theta = center_theta - np.deg2rad(self.range_margin)
            end_theta = center_theta + np.deg2rad(self.range_margin)



        for theta in np.arange(starting_theta, end_theta, dtheta):

            e1 = np.array([np.cos(theta), np.sin(theta)])
            e2 = np.array([-np.sin(theta), np.cos(theta)])

            c1 = X @ e1.T
            c2 = X @ e2.T

            cost = 0.0
            # Select criteria
            if self.criteria == self.Criteria.AREA:
                cost = self._calc_area_criterion(c1, c2)
            elif self.criteria == self.Criteria.CLOSENESS:
                cost = self._calc_closeness_criterion(c1, c2)
            elif self.criteria == self.Criteria.VARIANCE:
                cost = self._calc_variance_criterion(c1, c2)
            else:
                print("Invalid criteria")


            if minp[0] < cost:
                minp = (cost, theta)

        # calculate best rectangle
        sin_s = np.sin(minp[1])
        cos_s = np.cos(minp[1])

        c1_s = X @ np.array([cos_s, sin_s]).T
        c2_s = X @ np.array([-sin_s, cos_s]).T

        rect = RectangleData()
        rect.a[0] = cos_s
        rect.b[0] = sin_s
        rect.c[0] = min(c1_s)
        rect.a[1] = -sin_s
        rect.b[1] = cos_s
        rect.c[1] = min(c2_s)
        rect.a[2] = cos_s
        rect.b[2] = sin_s
        rect.c[2] = max(c1_s)
        rect.a[3] = -sin_s
        rect.b[3] = cos_s
        rect.c[3] = max(c2_s)

        intersection_x_1 = (rect.b[0] * rect.c[1] - rect.b[1] * rect.c[0]) / (rect.a[1] * rect.b[0] - rect.a[0] * rect.b[1])
        intersection_y_1 = (rect.a[0] * rect.c[1] - rect.a[1] * rect.c[0]) / (rect.a[0] * rect.b[1] - rect.a[1] * rect.b[0])
        intersection_x_2 = (rect.b[2] * rect.c[3] - rect.b[3] * rect.c[2]) / (rect.a[3] * rect.b[2] - rect.a[2] * rect.b[3])
        intersection_y_2 = (rect.a[2] * rect.c[3] - rect.a[3] * rect.c[2]) / (rect.a[2] * rect.b[3] - rect.a[3] * rect.b[2])


        e_x = np.array([rect.a[0] / np.sqrt(rect.a[0] * rect.a[0] + rect.b[0] * rect.b[0]), rect.b[0] / np.sqrt(rect.a[0] * rect.a[0] + rect.b[0] * rect.b[0])])
        e_y = np.array([rect.a[1] / np.sqrt(rect.a[1] * rect.a[1] + rect.b[1] * rect.b[1]), rect.b[1] / np.sqrt(rect.a[1] * rect.a[1] + rect.b[1] * rect.b[1])])
        diagonal_vec = np.array([intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2])
        yaw = np.arctan2(e_x[1], e_x[0])

        bbox = BoundingBox()
        bbox.pose.position.x = (intersection_x_1 + intersection_x_2) / 2.0
        bbox.pose.position.y = (intersection_y_1 + intersection_y_2) / 2.0
        bbox.pose.position.z = 0.0

        quat = Rotation.from_euler('xyz', [0, 0, yaw], degrees=False)
        quat_arr = quat.as_quat()

        bbox.pose.orientation.x = quat_arr[0]
        bbox.pose.orientation.y = quat_arr[1]
        bbox.pose.orientation.z = quat_arr[2]
        bbox.pose.orientation.w = quat_arr[3]

        bbox.dimensions.x = np.abs(np.dot(e_x, diagonal_vec))
        bbox.dimensions.y = np.abs(np.dot(e_y, diagonal_vec))
        bbox.dimensions.z = template_object.dimensions.z
        bbox.label = template_object.label
        bbox.value = template_object.value


        return self.correct_bbox(bbox, template_object)
    # Extend or shrink the source bbox in order to match the target bbox
    #
    def correct_bbox(self, source_bbox, target_bbox):
        obj_transform = np.eye(4)
        src_obj_rot = Rotation.from_quat([source_bbox.pose.orientation.x,
                                          source_bbox.pose.orientation.y,
                                          source_bbox.pose.orientation.z,
                                          source_bbox.pose.orientation.w]).as_matrix()
        obj_transform[0:3, 0:3] = src_obj_rot
        obj_transform[0:3, 3] = np.array([source_bbox.pose.position.x, source_bbox.pose.position.y, source_bbox.pose.position.z])


        if source_bbox.label == Classification.PEDESTRIAN.value:

            if source_bbox.dimensions.x < ShapeRestrictions.PED_MIN_LENGTH.value or \
                    source_bbox.dimensions.y < ShapeRestrictions.PED_MIN_WIDTH.value:
                return None

            if source_bbox.dimensions.x > ShapeRestrictions.PED_MAX_WIDTH.value or\
                source_bbox.dimensions.y > ShapeRestrictions.PED_MAX_WIDTH.value:
                return None
        elif source_bbox.label == Classification.CYCLIST.value:

            if source_bbox.dimensions.x < ShapeRestrictions.CYCLIST_MIN_LENGTH.value or\
                source_bbox.dimensions.y < ShapeRestrictions.CYCLIST_MIN_LENGTH.value:
                return None

            if source_bbox.dimensions.x > ShapeRestrictions.CYCLIST_MAX_LENGTH.value or\
                source_bbox.dimensions.y > ShapeRestrictions.CYCLIST_MAX_LENGTH.value:
                return None

        elif source_bbox.label == Classification.CAR.value:

            if source_bbox.dimensions.x < ShapeRestrictions.CAR_MIN_LENGTH.value and\
                source_bbox.dimensions.y < ShapeRestrictions.CAR_MIN_WIDTH.value:
                return None

            if source_bbox.dimensions.x > ShapeRestrictions.CAR_MAX_LENGTH.value and\
                source_bbox.dimensions.y > ShapeRestrictions.CAR_MAX_WIDTH.value:
                return None

            if source_bbox.dimensions.x < ShapeRestrictions.CAR_MIN_LENGTH.value:
                object_front_side = np.array([source_bbox.dimensions.x, 0, 0, 1])
                object_front_side_in_world = np.dot(obj_transform, object_front_side)

                object_back_side = np.array([-source_bbox.dimensions.x, 0, 0, 1])
                object_back_side_in_world = np.dot(obj_transform, object_back_side)

                shift = ShapeRestrictions.CAR_MIN_LENGTH.value - source_bbox.dimensions.x
                shift_vec_in_world = np.array([0, 0, 0])

                # If front side of the object is the closest to the origin, shift the object to the back
                if np.linalg.norm(object_front_side_in_world) < np.linalg.norm(object_back_side_in_world):
                    shift_vec = np.array([-shift/2.0, 0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)
                else:
                    shift_vec = np.array([shift/2.0, 0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)


                source_bbox.pose.position.x += shift_vec_in_world[0]
                source_bbox.pose.position.y += shift_vec_in_world[1]

                source_bbox.dimensions.x = ShapeRestrictions.CAR_MIN_LENGTH.value
            elif source_bbox.dimensions.x > ShapeRestrictions.CAR_MAX_LENGTH.value:

                object_front_side = np.array([source_bbox.dimensions.x, 0, 0, 1])
                object_front_side_in_world = np.dot(obj_transform, object_front_side)

                object_back_side = np.array([-source_bbox.dimensions.x, 0, 0, 1])
                object_back_side_in_world = np.dot(obj_transform, object_back_side)

                shift = ShapeRestrictions.CAR_MAX_LENGTH.value - source_bbox.dimensions.x
                shift_vec_in_world = np.array([0, 0, 0])

                # If front side of the object is the closest to the origin, shift the object to the back
                if np.linalg.norm(object_front_side_in_world) < np.linalg.norm(object_back_side_in_world):
                    shift_vec = np.array([shift/2.0, 0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)
                else:
                    shift_vec = np.array([-shift/2.0, 0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)

                source_bbox.pose.position.x += shift_vec_in_world[0]
                source_bbox.pose.position.y += shift_vec_in_world[1]

                source_bbox.dimensions.x = ShapeRestrictions.CAR_MAX_LENGTH.value

            if source_bbox.dimensions.y < ShapeRestrictions.CAR_MIN_WIDTH.value:

                # Check which side is closer to the origin
                object_left_side = np.array([0.0, source_bbox.dimensions.y, 0, 1])
                object_left_side_in_world = np.dot(obj_transform, object_left_side)

                object_right_side = np.array([0.0, -source_bbox.dimensions.y, 0, 1 ])
                object_right_side_in_world = np.dot(obj_transform, object_right_side)

                shift = ShapeRestrictions.CAR_MIN_WIDTH.value - source_bbox.dimensions.y
                shift_vec_in_world = np.array([0, 0, 0])

                # If left side of the object is the closest to the origin, shift the object to the right
                if np.linalg.norm(object_left_side_in_world) < np.linalg.norm(object_right_side_in_world):
                    shift_vec = np.array([0, -shift/2.0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)
                else:
                    shift_vec = np.array([0, shift/2.0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)


                source_bbox.pose.position.x += shift_vec_in_world[0]
                source_bbox.pose.position.y += shift_vec_in_world[1]

                source_bbox.dimensions.y = ShapeRestrictions.CAR_MIN_WIDTH.value

            elif source_bbox.dimensions.y > ShapeRestrictions.CAR_MAX_WIDTH.value:


                # Check which side is closer to the origin
                object_left_side = np.array([0.0, source_bbox.dimensions.y, 0, 1])
                object_left_side_in_world = np.dot(obj_transform, object_left_side)

                object_right_side = np.array([0.0, -source_bbox.dimensions.y, 0, 1 ])
                object_right_side_in_world = np.dot(obj_transform, object_right_side)


                shift = ShapeRestrictions.CAR_MAX_WIDTH.value - source_bbox.dimensions.y
                shift_vec_in_world = np.array([0, 0, 0])

                # If left side of the object is the closest to the origin, shift the object to the right
                if np.linalg.norm(object_left_side_in_world) < np.linalg.norm(object_right_side_in_world):
                    shift_vec = np.array([0, shift/2.0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)
                else:
                    shift_vec = np.array([0, -shift/2.0, 0])
                    shift_vec_in_world = np.dot(obj_transform[:3,:3], shift_vec)

                source_bbox.pose.position.x += shift_vec_in_world[0]
                source_bbox.pose.position.y += shift_vec_in_world[1]

                source_bbox.dimensions.y = ShapeRestrictions.CAR_MAX_WIDTH.value

        return source_bbox


    def _adoptive_range_segmentation(self, ox, oy):

        # Setup initial cluster
        S = []
        for i, _ in enumerate(ox):
            C = set()
            R = self.R0 + self.Rd * np.linalg.norm([ox[i], oy[i]])
            for j, _ in enumerate(ox):
                d = np.sqrt((ox[i] - ox[j])**2 + (oy[i] - oy[j])**2)
                try:
                    if d <= R:
                        C.add(j)
                except:
                    print("Error")
            S.append(C)

        # Merge cluster
        while 1:
            no_change = True
            for (c1, c2) in list(itertools.permutations(range(len(S)), 2)):
                if S[c1] & S[c2]:
                    S[c1] = (S[c1] | S.pop(c2))
                    no_change = False
                    break
            if no_change:
                break

        return S


class RectangleData():

    def __init__(self):
        self.a = [None] * 4
        self.b = [None] * 4
        self.c = [None] * 4

        self.rect_c_x = [None] * 5
        self.rect_c_y = [None] * 5

    def plot(self):
        self.calc_rect_contour()
        plt.plot(self.rect_c_x, self.rect_c_y, "-r")

    def calc_rect_contour(self):

        self.rect_c_x[0], self.rect_c_y[0] = self.calc_cross_point(self.a[0:2], self.b[0:2], self.c[0:2])

        self.rect_c_x[1], self.rect_c_y[1] = self.calc_cross_point(self.a[1:3], self.b[1:3], self.c[1:3])

        self.rect_c_x[2], self.rect_c_y[2] = self.calc_cross_point(self.a[2:4], self.b[2:4], self.c[2:4])

        self.rect_c_x[3], self.rect_c_y[3] = self.calc_cross_point([self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])

        self.rect_c_x[4], self.rect_c_y[4] = self.rect_c_x[0], self.rect_c_y[0]

    def calc_cross_point(self, a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y