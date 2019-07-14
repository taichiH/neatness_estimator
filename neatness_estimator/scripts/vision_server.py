#!/usr/bin/env python
# coding: UTF-8

import sys
import numpy as np

import rospy
import tf

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from neatness_estimator_msgs.srv import VisionServer, VisionServerResponse

class NeatnessEstimatorVisionServer():

    def __init__(self):
        self.label_lst = rospy.get_param('~fg_class_names')
        self.boxes = BoundingBoxArray()
        self.labeled_boxes = BoundingBoxArray()
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.shelf_flont_angle = 181.28271996356708

        rospy.Subscriber(
            "~input_instance_boxes", BoundingBoxArray, self.instance_box_callback)
        rospy.Subscriber(
            "~input_cluster_boxes", BoundingBoxArray, self.cluster_box_callback)
        rospy.Service(
            '/display_task_vision_server', VisionServer, self.vision_server)

    def instance_box_callback(self, msg):
        self.boxes = msg

    def cluster_box_callback(self, msg):
        self.labeled_boxes = msg


    ''' task get_obj_pos '''
    def get_obj_pos(self, req):
        print('get_obj_pos')

        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False
        has_nearest_item = False

        try:
            nearest_box, has_nearest_item = self.get_nearest_box(req)
            rospy.loginfo('has_nearest_item = true')
            if has_nearest_item:
                if req.parent_frame == '':
                    rospy.loginfo('req.parent_frame is: empty')
                    transformed_box = nearest_box
                else:
                    rospy.loginfo('req.parent_frame is: %s' %(self.boxes.header.frame_id))
                    transformed_box = self.transform_poses(
                        nearest_box.pose, req.label, self.boxes.header.frame_id, req.parent_frame)

                # faile lookup transform
                if transformed_box == BoundingBox():
                    transformed_box = nearest_box
                    transformed_box.header = self.boxes.header
                else:
                    transformed_box.header = self.boxes.header
                    transformed_box.header.frame_id = req.parent_frame
                    transformed_box.dimensions = nearest_box.dimensions

                res.boxes = transformed_box
                res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_distance_from_shelf_front '''
    def get_distance_from_shelf_front(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()

        try:
            has_shelf = False
            for box in self.boxes.boxes:
                if box.label == int(self.label_lst.index('shelf_flont')):
                    has_shelf = True
                    shelf_front = np.array([box.pose.position.x, box.pose.position.y])
                    shelf_edge = np.array([box.pose.position.x, box.pose.position.y]) +\
                                 np.array([box.dimensions.x * 0.5, box.dimensions.y * 0.5])

            if not has_shelf:
                rospy.logwarn("cannot find shelf front")
                res.status = False
                return res

            nearest_box, has_nearest_item = self.get_nearest_box(req)
            if has_nearest_item:
                target_vec = np.array([nearest_box.pose.position.x, nearest_box.pose.position.y])
                target_vec = target_vec - shelf_front
                shelf_vec = shelf_edge - shelf_front
                theta = np.arccos(
                    np.dot(target_vec, shelf_vec) / np.linalg.norm(target_vec) * np.linalg.norm(shelf_vec))
                theta = np.deg2rad(180) - theta if theta > np.deg2rad(90) else theta
                distance  = np.linalg.norm(target_vec) * np.sin(theta)

                if shelf_front is not None and target_vec is not None:
                    res.pulling_dist = distance
                    res.status = True
                else:
                    res.status = False
            else:
                res.status = False
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_distance_from_shelf_front_simple '''
    def get_distance_from_shelf_front_simple(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()

        try:
            has_shelf = False
            for box in self.boxes.boxes:
                if box.label == int(self.label_lst.index('shelf_flont')):
                    has_shelf = True
                    shelf_front = box.pose.position.x

            if not has_shelf:
                rospy.logwarn("cannot find shelf front")
                res.status = False
                return res

            nearest_box, has_nearest_item = self.get_nearest_box(req)
            if has_nearest_item:
                target = nearest_box.pose.position.x - box.dimensions.x * 0.5
                distance  = target - shelf_front

                if shelf_front is not None and target is not None:
                    res.pulling_dist = distance
                    res.status = True
                else:
                    res.status = False
            else:
                res.status = False
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res


    ''' task get_distance_between_two_items '''
    def get_items_distance(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()

        has_item = False
        has_ref_item = False

        try:
            left_side = -sys.maxsize # left side of right item
            right_side = sys.maxsize # right side of left item

            for box in self.boxes.boxes:
                if box.pose.position.x == 0 and box.pose.position.y == 0 and box.pose.position.z == 0:
                    continue

                if box.label == int(self.label_lst.index(req.ref_label)):
                    has_ref_item = True
                    tmp_left_side = box.pose.position.y + box.dimensions.y * 0.5
                    if tmp_left_side > left_side:
                        left_side = tmp_left_side

            for box in self.boxes.boxes:
                if box.label == int(self.label_lst.index(req.label)):
                    has_item = True
                    tmp_right_side = box.pose.position.y - box.dimensions.y * 0.5
                    if tmp_right_side < right_side:
                        right_side = tmp_right_side

            if right_side == sys.maxsize or left_side == -sys.maxsize \
               or not has_item or not has_ref_item:
                res.status = False
                res.filling_dist = 0
                return res

            res.filling_dist = right_side - left_side
            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_empty_space '''
    def get_empty_space(self, req):
        rospy.loginfo(req.task)
        rospy.logerr('get_empty_space is not used')
        res.status = False
        return res

    ''' task get_shelf_map_rotation '''
    def get_shelf_map_rotation(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        # overwrite request label
        req.label = 'shelf_flont'

        get_obj_pos_res = self.get_obj_pos(req)
        if get_obj_pos_res.status == True:

            qua = (get_obj_pos_res.boxes.pose.orientation.x,
                   get_obj_pos_res.boxes.pose.orientation.y,
                   get_obj_pos_res.boxes.pose.orientation.z,
                   get_obj_pos_res.boxes.pose.orientation.w)
            e = tf.transformations.euler_from_quaternion(qua)
            # print('rpy')
            # print(np.rad2deg(e[0]), np.rad2deg(e[1]), np.rad2deg(e[2]))

            res.boxes.pose.position.x = e[0]
            res.boxes.pose.position.y = e[1]
            res.boxes.pose.position.z = e[2]

            print(np.rad2deg(e[2]))

            if np.rad2deg(e[2]) == 0:
                res.status == False
                return res

            if np.rad2deg(e[2]) < 0:
                res.boxes.pose.orientation.z = -self.shelf_flont_angle - np.rad2deg(e[2])
            else:
                res.boxes.pose.orientation.z = self.shelf_flont_angle - np.rad2deg(e[2])

            print('angle diff: ', res.boxes.pose.orientation.z)
            res.status = True

        return res

    def get_nearest_box(self, req):
        distance = 100
        has_request_item = False
        nearest_box = BoundingBox()
        for index, box in enumerate(self.boxes.boxes):
            if self.label_lst[box.label] == req.label:
                has_request_item = True
                ref_point = np.array([box.pose.position.x + (box.dimensions.x * 0.5),
                                      box.pose.position.y + (box.dimensions.y * 0.5),
                                      box.pose.position.z + (box.dimensions.z * 0.5)])
                target_point = np.array([req.target.x,
                                         req.target.y,
                                         req.target.z])
                if np.linalg.norm(ref_point - target_point) < distance:
                    nearest_box.pose = box.pose
                    nearest_box.dimensions = box.dimensions
                    distance = np.linalg.norm(ref_point - target_point)

        return nearest_box, has_request_item

    def listen_transform(self, parent_frame, child_frame):
        box = BoundingBox()
        try:
            self.listener.waitForTransform(
                parent_frame, child_frame, rospy.Time(0), rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform(
                parent_frame, child_frame, rospy.Time(0))
            box.pose.position = Point(trans[0], trans[1], trans[2])
            box.pose.orientation = Quaternion(rot[0], rot[1], rot[2], rot[3])
            return box
        except:
            rospy.logwarn('cannot lookup transform')
            return box

    def transform_poses(self, pose, label, frame_id, parent):
        self.broadcaster.sendTransform(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            rospy.Time.now(), label, frame_id)
        box =  self.listen_transform(parent, label)
        return box

    def vision_server(self, req):
        if req.task == 'get_obj_pos':
            return self.get_obj_pos(req)

        elif req.task == 'get_distance':
            rospy.loginfo(req.task)
            return self.get_distance_from_shelf_front_simple(req)

        elif req.task == 'get_items_distance':
            rospy.loginfo(req.task)
            return self.get_items_distance(req)

        elif req.task == 'get_empty_space':
            rospy.loginfo(req.task)
            return self.get_empty_space(req)

        elif req.task == 'get_shelf_map_rotation':
            rospy.loginfo(req.task)
            return self.get_shelf_map_rotation(req)

if __name__ == "__main__":
    rospy.init_node("display_task_vision_server")
    vision_server = NeatnessEstimatorVisionServer()
    rospy.spin()
