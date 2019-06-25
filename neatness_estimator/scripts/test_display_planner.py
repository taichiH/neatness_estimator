#!/usr/bin/env python

import sys
import numpy as np
import copy

import rospy
from neatness_estimator_msgs.srv import DisplayState, UpdateBoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import Pose, Point, PoseStamped

class TestDisplayPlanner():

    def __init__(self):
        self.res = None
        self.init_boxes = BoundingBoxArray()
        self.dynamic_boxes = BoundingBoxArray()
        self.check_callback = True
        self.tmp_space_pos_list = None
        self.reference_state = [12,12,12,12,4,4,4,4,16,16,16,16]
        self.border_indexes = [0,4,8,12]
        self.update_boxes_client = rospy.ServiceProxy(
            '/dummy_bounding_box_publisher/update_boxes', UpdateBoundingBox)

        self.debug_pose_pub = rospy.Publisher('~debug_pose', PoseStamped, queue_size=1)
        rospy.Subscriber(
            '/dummy_bounding_box_publisher/output', BoundingBoxArray, self.box_array_callback)

    def box_array_callback(self, boxes_msg):
        for box in boxes_msg.boxes:
            if box.label == 17:
                continue
            self.dynamic_boxes.boxes.append(box)

        self.frame_id = boxes_msg.header.frame_id
        if self.check_callback:
            self.init_boxes = boxes_msg.boxes
            self.tmp_space_pos_list = [
                Point(0.0 + i * 0.05, 0, 0.05) for i in range(len(boxes_msg.boxes))]
            self.run()
            self.check_callback = False

    def run(self):
        display_client = rospy.ServiceProxy('/display_planner_server', DisplayState)
        self.res = display_client(
            reference_state=self.reference_state, border_indexes=self.border_indexes)
        self.visualize()

    def calc_ignore_indexes(self, index):
        ignore_indexes = []
        for i in range(1, len(self.border_indexes)):
            if index in range(self.border_indexes[i-1], self.border_indexes[i]):
                ignore_indexes = range(self.border_indexes[i-1], self.border_indexes[i])
        return ignore_indexes

    def get_nearest_box_index(self, input_box):
        min_norm = sys.maxsize
        min_index = 0

        for i, box in enumerate(self.init_boxes):
            ref_box = np.array(
                [box.pose.position.x, box.pose.position.y, box.pose.position.z])
            np_box = np.array(
                [input_box.pose.position.x, input_box.pose.position.y, input_box.pose.position.z])
            distance = np.linalg.norm(ref_box - np_box)
            if distance < min_norm:
                min_index = i
                min_norm = distance
        return min_index

    def get_nearest_reference_box_index(self, input_box, current_index):
        min_norm = sys.maxsize
        min_index = 0

        ignore_indexes = self.calc_ignore_indexes(current_index)

        for i, box in enumerate(self.dynamic_boxes.boxes):
            if box.label != input_box.label:
                continue

            if i in ignore_indexes:
                continue

            ref_box = np.array(
                [box.pose.position.x, box.pose.position.y, box.pose.position.z])
            np_box = np.array(
                [input_box.pose.position.x, input_box.pose.position.y, input_box.pose.position.z])
            distance = np.linalg.norm(ref_box - np_box)
            if distance < min_norm and i not in ignore_indexes:
                min_index = i
                min_norm = distance
        return min_index

    def save_to_tmp_space(self, index):
        if self.tmp_space_pos_list is None:
            return False
        self.dynamic_boxes.boxes[index].pose.position = self.tmp_space_pos_list.pop()

    def update_boxes(self, current_index, reference_index):
        tmp_pose = copy.copy(self.dynamic_boxes.boxes[current_index].pose)
        self.save_to_tmp_space(current_index)
        print('current_index in update_boxes', current_index)
        self.update_boxes_client(boxes=self.get_client_boxes_msg())

        print('--- update_boxes ---')
        raw_input()

        self.dynamic_boxes.boxes[reference_index].pose = tmp_pose

        debug_pose_msg = PoseStamped()
        debug_pose_msg.pose = self.dynamic_boxes.boxes[reference_index].pose
        debug_pose_msg.header.frame_id = self.frame_id
        debug_pose_msg.header.stamp = rospy.Time.now()
        self.debug_pose_pub.publish(debug_pose_msg)

    def get_client_boxes_msg(self):
        client_boxes_msg = BoundingBoxArray()
        client_boxes_msg.boxes = self.dynamic_boxes.boxes
        client_boxes_msg.header.stamp = rospy.Time.now()
        client_boxes_msg.header.frame_id = self.frame_id
        return client_boxes_msg

    def visualize(self):
        print(self.res.distance)
        if self.res.status:
            for i, task in enumerate(self.res.plan.tasks):
                print('step_%s' %(str(i)))
                current_index = self.get_nearest_box_index(task.current)
                reference_index = self.get_nearest_reference_box_index(
                    task.reference, current_index)
                self.update_boxes(current_index, reference_index)

                self.update_boxes_client(boxes=self.get_client_boxes_msg())
                print('--- res.plan.tasks ---')
                raw_input()

if __name__=='__main__':
    rospy.init_node('test_display_planner')
    test_display_planner = TestDisplayPlanner()
    rospy.spin()
