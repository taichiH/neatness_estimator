#!/usr/bin/env python
# coding: UTF-8

import numpy as np

import rospy
import tf

try:
    from what_i_see_msgs.msg import LabeledPoseArray
except:
    rospy.logerr("please install negomo")
    exit()

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from neatness_estimator.srv import VisionServer, VisionServerResponse

class NeatnessVisionServer():

    def __init__(self):
        self.label_list = ["alfort", "alfortwhite", "almond", "apc_shelf_flont",\
                           "coffee", "consome", "dars", "darsmilk", "darswhite",\
                           "donbe", "kinoko", "macadamia", "milk", "mixjuice",\
                           "marble", "norishio", "pie", "shelf_flont", "takenoko",\
                           "tee", "xylitop", "yakisoba"]
        self.boxes = LabeledPoseArray()
        self.labeled_boxes = BoundingBoxArray()

        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()

        rospy.Subscriber("/mask_rcnn_projector/output",
                         LabeledPoseArray,
                         self.mask_rcnn_callback)
        rospy.Subscriber("/labeled_bounding_box_publisher/output/labeled_cluster_boxes",
                         BoundingBoxArray,
                         self.slide_adjust_callback)
        rospy.Service('/neatness_vision_server',
                      VisionServer,
                      self.vision_server)

    def mask_rcnn_callback(self, msg):
        self.boxes = msg

    def slide_adjust_callback(self, msg):
        self.labeled_boxes = msg

    def vision_server(self, req):
        if req.task == 'get_obj_pos':
            rospy.loginfo(req.task)
            distance = 100
            nearest_box = BoundingBox()
            for index, box in enumerate(self.boxes.poses):
                if box.label == req.label:
                    ref_point = np.array([box.pose.position.x,
                                      box.pose.position.y,
                                      box.pose.position.z])
                    target_point = np.array([req.target.x,
                                         req.target.y,
                                         req.target.z])
                    if np.linalg.norm(ref_point - target_point) < distance:
                        nearest_box.pose = box.pose
                        distance = np.linalg.norm(ref_point - target_point)

            res = VisionServerResponse()
            transformed_box = self.transform_poses(nearest_box.pose,
                                                   req.label,
                                                   self.boxes.header.frame_id,
                                                   req.parent_frame)
            transformed_box.header = self.boxes.header
            transformed_box.header.frame_id = req.parent_frame
            res.boxes = transformed_box
            res.status = True
            return res

        elif req.task == 'get_dist':
            rospy.loginfo(req.task)
            target_left = 0
            for box in self.labeled_boxes.boxes:
                if box.label == int(self.label_list.index(req.label)):
                    target_left = box.pose.position.y + box.dimensions.y * 0.5
                    break

            min_dist = 24 ** 2
            for box in self.labeled_boxes.boxes:
                ref_right = box.pose.position.y - box.dimensions.y * 0.5
                if abs(ref_right - target_left) < min_dist:
                    min_dist = abs(ref_right - target_left)

            res = VisionServerResponse()
            res.dist = min_dist
            res.status = True
            return res

        elif req.task == 'get_empty_space':
            rospy.loginfo(req.task)
            for box in self.labeled_boxes.boxes:
                if box.label == int(self.label_list.index(req.label)):
                    shelf_lt = np.array([box.pose.position.x + box.dimensions.x * 0.5,
                                         box.pose.position.y + box.dimensions.y * 0.5,
                                         box.pose.position.z + box.dimensions.z * 0.5])
                    shelf_rb = np.array([box.pose.position.x - box.dimensions.x * 0.5,
                                         box.pose.position.y - box.dimensions.y * 0.5,
                                         box.pose.position.z - box.dimensions.z * 0.5])
                    break

            cliped_bounding_box = BoundingBoxArray()
            offset = 0.15

            for box in self.labeled_boxes.boxes:
                if box.pose.position.y < (shelf_lt[1] + offset) and \
                   box.pose.position.y > (shelf_rb[1] - offset) and \
                   box.pose.position.z > (shelf_lt[2] - offset):
                    print(box)
                    cliped_bounding_box.boxes.append(box)

            sorted_bounding_box = BoundingBoxArray()
            sorted_bounding_box.boxes = sorted(cliped_bounding_box.boxes,
                                               key=lambda box : box.pose.position.y)

            max_dist = 0
            index = 0
            for i in range(1, len(sorted_bounding_box.boxes)):
                dist = (sorted_bounding_box.boxes[i].pose.position.y - \
                        sorted_bounding_box.boxes[i].dimensions.y * 0.5) - \
                        (sorted_bounding_box.boxes[i-1].pose.position.y + \
                         sorted_bounding_box.boxes[i-1].dimensions.y * 0.5)
                if dist > max_dist:
                    index = i
                    max_dist = dist

            print('index: ', index)
            res = VisionServerResponse()
            empty_space_box = BoundingBox()
            empty_space_box.header = self.labeled_boxes.header

            empty_space_box.pose.position.x = (sorted_bounding_box.boxes[index].pose.position.x + \
                                               sorted_bounding_box.boxes[index-1].pose.position.x) * 0.5
            empty_space_box.pose.position.y = (sorted_bounding_box.boxes[index].pose.position.y + \
                                               sorted_bounding_box.boxes[index-1].pose.position.y) * 0.5
            empty_space_box.pose.position.z = (sorted_bounding_box.boxes[index].pose.position.z + \
                                               sorted_bounding_box.boxes[index-1].pose.position.z) * 0.5
            empty_space_box.pose.orientation = sorted_bounding_box.boxes[index].pose.orientation
            empty_space_box.dimensions.x = (sorted_bounding_box.boxes[index].dimensions.x + \
                                            sorted_bounding_box.boxes[index-1].dimensions.x) * 0.5
            empty_space_box.dimensions.y = max_dist
            empty_space_box.dimensions.z = (sorted_bounding_box.boxes[index].dimensions.z + \
                                            sorted_bounding_box.boxes[index-1].dimensions.z) * 0.5

            # transform map -> base_link
            transformed_box = self.transform_poses(empty_space_box.pose,
                                                   "empty_space",
                                                   empty_space_box.header.frame_id,
                                                   req.parent_frame)
            empty_space_box.pose = transformed_box.pose
            res.boxes = empty_space_box
            res.status = True
            return res

        else:
            rospy.loginfo('not much task')
            res = VisionServerResponse()
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

if __name__ == "__main__":
    rospy.init_node("neatness_vision_server")
    nvs = NeatnessVisionServer()
    rospy.spin()
