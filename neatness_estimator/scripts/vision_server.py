#!/usr/bin/env python
# coding: UTF-8

import numpy as np

import rospy
import tf

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from neatness_estimator_msgs.srv import VisionServer, VisionServerResponse

class NeatnessEstimatorVisionServer():

    def __init__(self):
        self.label_lst = rospy.get_param('~fg_class_names')
        self.boxes = BoundingBoxArray()
        self.labeled_boxes = BoundingBoxArray()

        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()

        rospy.Subscriber("~input_instance_boxes",
                         BoundingBoxArray,
                         self.instance_box_callback)
        rospy.Subscriber("~input_cluster_boxes",
                         BoundingBoxArray,
                         self.cluster_box_callback)
        rospy.Service('/display_task_vision_server',
                      VisionServer,
                      self.vision_server)

    def instance_box_callback(self, msg):
        self.boxes = msg

    def cluster_box_callback(self, msg):
        self.labeled_boxes = msg

    def get_obj_pos(self, req):
        rospy.loginfo(req.task)
        distance = 100
        nearest_box = BoundingBox()
        for index, box in enumerate(self.boxes.boxes):
            if self.label_lst[box.label] == req.label:
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

        res = VisionServerResponse()
        transformed_box = self.transform_poses(nearest_box.pose,
                                               req.label,
                                               self.boxes.header.frame_id,
                                               req.parent_frame)
        transformed_box.header = self.boxes.header
        transformed_box.header.frame_id = req.parent_frame
        transformed_box.dimensions = nearest_box.dimensions
        res.boxes = transformed_box
        res.status = True
        return res

    def get_distance_from_shelf_front(self, req):
        shelf_front = None
        target = None
        for box in self.boxes.boxes:
            if box.label == int(self.label_lst.index('shelf_flont')):
                shelf_front = np.array([box.pose.position.x,
                                        box.pose.position.y,
                                        box.pose.position.z])
            if box.label == int(self.label_lst.index(req.label)):
                target = np.array([box.pose.position.x,
                                   box.pose.position.y,
                                   box.pose.position.z])

        res = VisionServerResponse()
        if shelf_front is not None and target is not None:
            res.pulling_dist = np.linalg.norm(shelf_front - target)
            res.status = True
        else:
            res.status = False
        return res

    def get_empty_space(self, req):
        rospy.loginfo(req.task)
        rospy.logerr('get_empty_space is not used')
        res.status = False
        return res

    def listen_transform(self, parent_frame, child_frame):
        box = BoundingBox()
        try:
            self.listener.waitForTransform(parent_frame,
                                           child_frame,
                                           rospy.Time(0),
                                           rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform(parent_frame,
                                                         child_frame,
                                                         rospy.Time(0))
            box.pose.position.x = trans[0]
            box.pose.position.y = trans[1]
            box.pose.position.z = trans[2]
            box.pose.orientation.x = rot[0]
            box.pose.orientation.y = rot[1]
            box.pose.orientation.z = rot[2]
            box.pose.orientation.w = rot[3]
            return box
        except:
            rospy.logwarn('cannot lookup transform')
            return box

    def transform_poses(self, pose, label, frame_id, parent):
        self.broadcaster.sendTransform((pose.position.x,
                                        pose.position.y,
                                        pose.position.z),
                                       (pose.orientation.x,
                                        pose.orientation.y,
                                        pose.orientation.z,
                                        pose.orientation.w),
                                       rospy.Time.now(),
                                       label,
                                       frame_id)

        box =  self.listen_transform(parent, label)
        return box

    def vision_server(self, req):
        if req.task == 'get_obj_pos':
            return self.get_obj_pos(req)

        elif req.task == 'get_distance':
            rospy.loginfo(req.task)
            return self.get_distance_from_shelf_front(req)

        elif req.task == 'get_empty_space':
            rospy.loginfo(req.task)
            return self.get_empty_space(req)

if __name__ == "__main__":
    rospy.init_node("display_task_vision_server")
    vision_server = NeatnessEstimatorVisionServer()
    rospy.spin()
