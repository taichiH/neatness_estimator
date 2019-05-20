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

class DisplayPlanner():

    def __init__(self):
        self.boxes = BoundingBoxArray()
        self.sorted_boxes = BoundingBoxArray()
        self.border_indexes = None

        self.change = rospy.get_param('change', 7)
        self.insert = rospy.get_param('insert', 14)
        self.delete = rospy.get_param('delete', 14)

        rospy.Subscriber(
            "~input_boxes", BoundingBoxArray, self.callback)
        rospy.Service(
            '/display_planner_server', DisplayState, self.server_callback)

    def callback(self, msg):
        self.boxes = msg

    def server_callback(self, req):
        self.border_indexes = req.border_indexes
        reference_state = map(str, req.reference_state)
        current_state = self.boxes_to_state(self.boxes)

        table = self.initialize_table(current_state, reference_state)
        calculated_table = self.calculate_cost(table, current_state, reference_state)
        plan, edit = self.calculate_plan(calculated_table, current_state, reference_state)

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

    def create_plan(self, plan, cost):
        display_plans = DisplayPlanArray()
        res = DisplayStateResponse()

        for i, task in enumerate(plan):
            if task[1] == 'replace':
                print(task)
                display_plan = DisplayPlan()
                display_plan.current = self.sorted_boxes.boxes[i]
                display_plan.current.label = int(task[0][0])
                display_plan.reference = self.get_reference_box(i, int(task[0][1]), display_plan.current)
                display_plan.manipulation = 'replace'
                display_plans.tasks.append(display_plan)

        display_plans.header = self.boxes.header

        res.plan = display_plans
        res.status = True
        res.distance = cost
        return res

    def get_reference_box(self, index, ref_index, current):
        # ex: [0,6,8]
        ignore_indexes = []
        for i in range(1, len(self.border_indexes)):
            if index in range(self.border_indexes[i-1], self.border_indexes[i]):
                ignore_indexes = range(self.border_indexes[i-1], self.border_indexes[i])

        norm = 2 ** 24
        current_vec = np.array(
            [current.pose.position.x, current.pose.position.y, current.pose.position.z])

        for i, box in enumerate(self.sorted_boxes.boxes):
            if i in ignore_indexes or box.label != ref_index:
                continue
            distance = np.linalg.norm(
                np.array([box.pose.position.x, box.pose.position.y, box.pose.position.z]) - current_vec)
            if distance < norm:
                norm = distance
                reference_box = box

        return reference_box

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
