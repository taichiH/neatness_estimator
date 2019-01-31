#!/usr/bin/env python
# coding: UTF-8
import os
import numpy as np
import rospy
import rospkg
import tf
import message_filters
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_recognition_msgs.msg import ClassificationResult
from neatness_estimator_msgs.msg import Neatness
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from distance_clustering import Clustering
from neatness_estimator_msgs.msg import TargetAndGoal, TargetAndGoalArray

from visualization_msgs.msg import Marker

class NeatnessEstimator():

    def __init__(self):
        self.label_lst = rospy.get_param('~fg_class_names')
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        self.box_pub = rospy.Publisher('~output_box',
                                       BoundingBoxArray,
                                       queue_size=1)
        self.neatness_pub = rospy.Publisher('~output_neatness',
                                           Neatness,
                                           queue_size=1)
        self.marker_pub = rospy.Publisher('~output_marker',
                                          Marker,
                                          queue_size=1)
        self.target_and_goals_pub = rospy.Publisher('~target_and_goals',
                                                    TargetAndGoalArray,
                                                    queue_size=1)

        self.clustering = Clustering()
        self.row_nums = 2
        self.second_cluster_limit = rospy.get_param('~second_cluster_limit', 0.04)
        self.thresh = rospy.get_param('~thresh', 0.8)
        self.target_object = 'mixjuice'
        self.target = None
        self.goal = None
        self.boxes = []
        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_instance_box = message_filters.Subscriber(
            '~input/instance_boxes',
            BoundingBoxArray, queue_size=1, buff_size=2**24)

        self.subs = [sub_instance_box]

        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.callback)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def callback(self, instance_msg):
        labeled_boxes = {}
        label_buf = []

        for box in instance_msg.boxes:
            if box.label in label_buf:
                labeled_boxes[box.label] += [self.get_points(box)]
                orientation = box.pose.orientation
            else:
                labeled_boxes[box.label] = [self.get_points(box)]
                label_buf.append(box.label)

        bounding_box_msg = BoundingBoxArray()
        target_and_goal_msg = TargetAndGoal()
        for label, boxes in zip(labeled_boxes.keys(), labeled_boxes.values()):
            if self.label_lst[label] != self.target_object:
                continue
            thresh = self.thresh
            if self.label_lst[label] == 'shelf_flont':
                thresh = 2.0
            boxes = np.array(boxes)
            result = self.clustering.clustering_wrapper(boxes, thresh)
            transformed_pos = []
            for cluster in enumerate(result):                
                print('-------------- %s' %(self.label_lst[label]))
                transformed_pos = []
                faile_transform = False
                for j in cluster.indices:
                    name = 'item_' + str(j)
                    (trans, rot) = self.transform(boxes[j],
                                                  name,
                                                  instance_msg.header)
                    if len(trans) == 0 and len(rot) == 0:
                        faile_transform = True
                    transformed_pos.append(np.array(trans))

                if faile_transform:
                    rospy.logwarn('listen transform failed')
                    continue
                sorted_pos = sorted(transformed_pos, key = lambda x : x[0])
                target_and_goals = gen_arrangement(len(sorted_pos),
                                                   self.row_nums,
                                                   sorted_pos,
                                                   instance_msg.header)
                bounding_box_msg.boxes.append(gen_bounding_box(boxes, i))

        bounding_box_msg.header = instance_msg.header
        self.box_pub.publish(bounding_box_msg)
        target_and_goals.header = instance_msg.header
        self.target_and_goals_pub.publish(target_and_goals)

    def gen_bounding_box(boxes, i):
        orientation = Pose().orientation
        max_candidates = [boxes[i][0] + (boxes[i][1] * 0.5) for i in cluster.indices]
        min_candidates = [boxes[i][0] - (boxes[i][1] * 0.5) for i in cluster.indices]
        candidates = np.array(max_candidates + min_candidates)
        dimension = candidates.max(axis=0) - candidates.min(axis=0)
        center = candidates.min(axis=0) + (dimension * 0.5)
        tmp_box = BoundingBox()
        tmp_box.header = instance_msg.header
        tmp_box.dimensions.x = dimension[0]
        tmp_box.dimensions.y = dimension[1]
        tmp_box.dimensions.z = dimension[2]
        tmp_box.pose.position.x = center[0]
        tmp_box.pose.position.y = center[1]
        tmp_box.pose.position.z = center[2]
        tmp_box.pose.orientation = orientation
        tmp_box.label = label
        return tmp_box

    def gen_arrangement(elem_num, row_num, sorted_pos, header):
        initial_pos = sorted_pos[0]
        stride = 1
        vec = np.zeros((3))

        target_and_goals = TargetAndGoalArray()
        for i in range(elem_num):
            target_and_goal = TargetAndGoal()
            if (i+1) % row_num == 0:
                vec[0] = 0.0
                vec[1] += stride
            goal = initial_pos + vec
            index = np.array([np.linalg.norm(goal - pos) for pos in sorted_pos]).argmin()
            target = sorted_pos[index]
            sorted_pos.pop(index)
            vec[0] += stride

            target_and_goal.target.point = Point(target[0], target[1], target[2])
            target_and_goal.target.header = header
            target_and_goal.goal.point = Point(goal[0], goal[1], goal[2])
            target_and_goal.goal.header = header
            target_and_goals.positions.append(target_and_goal)
        return target_and_goals

    def transform(self, box, name, header):
        self.broadcaster.sendTransform((box[0][0],
                                        box[0][1],
                                        box[0][2]),
                                       (1, 0, 0, 0),
                                       rospy.Time.now(),
                                       name,
                                       header.frame_id)
        try:
            self.listener.waitForTransform("base_link",
                                           name,
                                           rospy.Time(0),
                                           rospy.Duration(5.0))
            (trans, rot) = self.listener.lookupTransform("base_link",
                                                         name,
                                                         rospy.Time(0))
            return (trans, rot)
        except:            
            return ([], [])

    def get_points(self, box):
        return (np.array([box.pose.position.x,
                          box.pose.position.y,
                          box.pose.position.z,
                          box.dimensions.x,
                          box.dimensions.y,
                          box.dimensions.z])).reshape(2, 3)

    def visualize(self, a, b, header):
        marker = Marker()
        marker.header = header
        marker.ns = 'line'
        marker.id = 1
        marker.type = Marker.LINE_STRIP

        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.points.append(Point(a[0], a[1], a[2]))
        marker.points.append(Point(b[0], b[1], b[2]))
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('neatness_estimator')
    neatness_estimator = NeatnessEstimator()
    rospy.spin()
