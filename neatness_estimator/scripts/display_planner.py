#! /usr/bin/env python
# -*- coding: utf-8 -*-

# likelihood
# 1 - (edit distance / length of the larger of the two strings)

import collections
import numpy as np

import rospy
from neatness_estimator_msgs.srv import DisplayState, DisplayStateResponse
from neatness_estimator_msgs.msg import DisplayPlan, DisplayPlanArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Pose, Point, PoseStamped

class DisplayPlanner():

    def __init__(self):
        self.boxes = BoundingBoxArray()
        self.dynamic_boxes = BoundingBoxArray()
        self.sorted_boxes = BoundingBoxArray()
        self.border_indexes = None

        self.change = rospy.get_param('change', 7)
        self.insert = rospy.get_param('insert', 14)
        self.delete = rospy.get_param('delete', 14)

        self.tmp_space_pos_list = [
            Point(0.0 + i * 0.05, 0, 0.05) for i in range(16)]
        self.boxes_buffer = []
        self.ref_index_buffer = []

        self.debug_current_pose_pub = rospy.Publisher('~debug_current_pose', PoseStamped, queue_size=1)
        self.debug_ref_pose_pub = rospy.Publisher('~debug_ref_pose', PoseStamped, queue_size=1)

        self.update_boxes_client = rospy.ServiceProxy(
            '/dummy_bounding_box_publisher/update_boxes', UpdateBoundingBox)

        rospy.Subscriber(
            "~input_boxes", BoundingBoxArray, self.callback)
        rospy.Service(
            '/display_planner_server', DisplayState, self.server_callback)

    def callback(self, msg):
        self.boxes = msg
        # for box in msg.boxes:
        #     if box.label == 17:
        #         continue
        #     self.boxes.boxes.append(box)

    def server_callback(self, req):
        self.border_indexes = req.border_indexes

        reference_state = map(str, req.reference_state)
        current_state = self.boxes_to_state(self.boxes)
        table = self.initialize_table(current_state, reference_state)
        calculated_table = self.calculate_cost(table, current_state, reference_state)
        plan, edit = self.calculate_plan(calculated_table, current_state, reference_state)

        index = 0
        for i, box in enumerate(self.sorted_boxes.boxes):
            if box.label == 17:
                continue
            if i == self.border_indexes[index]:
                index += 1
            box.value = index
            self.dynamic_boxes.boxes.append(box)

        print('--------------------')
        print('current: ', current_state)
        print('reference: ', reference_state)
        print('border indexes: ', self.border_indexes)
        # for task in plan:
        #     print(task)

        # normalized edit distance means neatness
        cost = 1.0 - (edit / float(max(len(req.reference_state), len(current_state))))

        res = self.create_plan(plan, cost)
        return res

    def update_boxes(self, current_index, reference_index):
        return true

    def replace(self, index, task):
        ref_box, ref_index = get_reference_box(index, task)

        buffer_index = len(self.boxes_buffer)
        self.dynamic_boxes.boxes[index].pose = self.tmp_space_pos_list[buffer_index]
        self.boxes_buffer.append(dynamic_boxes.boxes[index])

        if ref_box is None:
            ref_box = self.boxes_buffer.pop()

        current_box = copy.copy(self.sorted_boxes.boxes[index])
        self.dynamic_boxes.boxes[ref_index].pose = current_box
        ref_index_buffer.append(ref_index)

        return display_plan

    def get_reference_box(self, index, task):
        # ex: [0,6,8]
        ignore_indexes = []
        for i in range(1, len(self.border_indexes)):
            if index in range(self.border_indexes[i-1], self.border_indexes[i]):
                ignore_indexes = range(self.border_indexes[i-1], self.border_indexes[i])

        norm = 2 ** 24
        current_vec = np.array(
            [current.pose.position.x, current.pose.position.y, current.pose.position.z])

        reference_box = None
        reference_index = None
        for i, box in enumerate(self.dynamic_boxes.boxes[index:]):
            if i in ignore_indexes or box.label != task[0][1]:
                continue

            distance = np.linalg.norm(
                np.array([box.pose.position.x, box.pose.position.y, box.pose.position.z]) - current_vec)
            if distance < norm:
                norm = distance
                reference_box = box
                reference_index = i

        return reference_box, reference_index

    def create_plan(self, plan, cost):
        display_plans = DisplayPlanArray()
        res = DisplayStateResponse()

        for i, task in enumerate(plan):
            if task[1] == 'replace':
                print(task)
                display_plan = self.replace(i, task)
                display_plans.tasks.append(display_plan)

        display_plans.header = self.boxes.header

        res.plan = display_plans
        res.status = True
        res.distance = cost
        return res


    def boxes_to_state(self, boxes):
        tmp_boxes = sorted(boxes.boxes, key=lambda box : box.pose.position.y, reverse=True)
        self.sorted_boxes = BoundingBoxArray()
        self.sorted_boxes.boxes = tmp_boxes

        current_state = []
        for box in self.sorted_boxes.boxes:
            if box.label == 17:
                continue
            current_state.append(str(box.label))
        return current_state

    def initialize_table(self, current, reference):
        table = [[(0,0,0)] * (len(current) + 1) for i in range(len(reference) + 1)]
        for col in range(1, len(table[0])):
            table[0][col] = (table[0][col - 1][0] + self.change, 0, col - 1)
        for row in range(1, len(table)):
            table[row][0] = (table[row - 1][0][0] + self.change, row - 1, 0)
        return table

    def calculate_cost(self, table, current, refference):
        for row in range(1, len(table)):
            for col in range(1, len(table[0])):
                if current[col - 1] == refference[row - 1]:
                    table[row][col] = (table[row - 1][col - 1][0], row - 1, col - 1)
                else:
                    up_left = (table[row - 1][col - 1][0] + self.change, row - 1, col - 1)
                    left = (table[row][col - 1][0] + self.delete, row, col - 1)
                    up = (table[row - 1][col][0] + self.insert, row - 1, col)
                    table[row][col] = sorted([up_left, left, up], key=lambda x: x[0])[0]
        return table

    def calculate_plan(self, table, current, reference):
        results = []
        follow = (len(current), len(reference))
        while follow != (0, 0):
            point = table[follow[0]][follow[1]]
            route = (point[1], point[2])

            if follow[0] == route[0]:
                results.append(([current[route[1]]], 'delete'))
            elif follow[1] == route[1]:
                results.append(([current[route[0]]], 'insert'))
            elif table[route[0]][route[1]][0] == point[0]:
                results.append(([current[route[1]]], 'match'))
            else:
                results.append(([current[route[1]], reference[route[0]]], 'replace'))
            follow = route

        results.reverse()

        edit = 0
        for result in results:
            if result[1] != 'match':
                edit += 1

        return results, edit

if __name__ == '__main__':
    rospy.init_node('display_planner')
    display_planner = DisplayPlanner()
    rospy.spin()
