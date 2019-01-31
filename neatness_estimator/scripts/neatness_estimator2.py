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
from neatness_estimator_msgs.msg import TargetAndGoal

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
        self.target_and_goal_pub = rospy.Publisher('~target_and_goal',
                                                   TargetAndGoal,
                                                   queue_size=1)

        self.clustering = Clustering()
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
        orientation = Pose().orientation

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

            second_thresh = 0.4
            step = 0.01

            max_distance = 0
            for i, cluster in enumerate(result):                
                print('-------------- %s' %(self.label_lst[label]))
                clustered_boxes = np.array([np.array(boxes[i]) for i in cluster.indices])
                if len(clustered_boxes) > 1:
                    second_result, distance = self.second_clustering(clustered_boxes, second_thresh, step)
                    if len(second_result) > 1:
                        transformed_pos = []
                        for devided_cluster in second_result:
                            pos = []
                            faile_transform = False
                            for j in devided_cluster.indices:
                                name = 'item_' + str(j)
                                (trans, rot) = self.transform(clustered_boxes[j], name, instance_msg.header)
                                if len(trans) == 0 and len(rot) == 0:
                                    rospy.logwarn('listen transform failed')
                                    faile_transform = True

                                pos.append(np.array(trans))
                            transformed_pos.append(pos)

                            if faile_transform:
                                continue

                        target, goal = self.get_target_and_goal(second_result, transformed_pos)

                        if target.sum() == 0 or \
                           goal.sum() == 0 or \
                           True in np.isnan(target) or \
                           True in np.isnan(goal):
                            rospy.logwarn('failed get target and goal')
                            continue

                        if distance > max_distance:
                            print(distance)
                            print(target,goal)
                            target_and_goal_msg.target.point = Point(target[0], target[1], target[2])
                            target_and_goal_msg.goal.point = Point(goal[0], goal[1], goal[2])
                            target_and_goal_msg.target.header = target_and_goal_msg.goal.header = instance_msg.header

                        
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
                bounding_box_msg.boxes.append(tmp_box)


        # print('broadcast target and goal')
        # print(target_and_goal_msg.target.point.x,
        #       target_and_goal_msg.target.point.y,
        #       target_and_goal_msg.target.point.z)
        # print(target_and_goal_msg.goal.point.x,
        #       target_and_goal_msg.goal.point.y,
        #       target_and_goal_msg.goal.point.z)

        self.broadcaster.sendTransform((target_and_goal_msg.target.point.x,
                                        target_and_goal_msg.target.point.y,
                                        target_and_goal_msg.target.point.z),
                                       (1, 0, 0, 0),
                                       rospy.Time.now(),
                                       "target",
                                       "base_link")
        self.broadcaster.sendTransform((target_and_goal_msg.goal.point.x,
                                        target_and_goal_msg.goal.point.y,
                                        target_and_goal_msg.goal.point.z),
                                       (1, 0, 0, 0),
                                       rospy.Time.now(),
                                       "goal",
                                       "base_link")

        bounding_box_msg.header = instance_msg.header
        self.box_pub.publish(bounding_box_msg)
        target_and_goal_msg.target.header.frame_id = target_and_goal_msg.goal.header.frame_id = "base_link"
        self.target_and_goal_pub.publish(target_and_goal_msg)


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

    def second_clustering(self, clustered_boxes, second_thresh, step):
        second_result = self.clustering.clustering_wrapper(clustered_boxes, second_thresh)
        if second_thresh < self.second_cluster_limit:
            return second_result, second_thresh

        # when cluster is devided into two
        if len(second_result) > 1:
            return second_result, second_thresh

        clustered_boxes = np.array([np.array(clustered_boxes[i]) for i in second_result[0].indices])
        return self.second_clustering(clustered_boxes, second_thresh - step, step)

    def get_target_and_goal(self, second_result, transformed_pos):
        # second_result index == tranformed_pos index
        if len(second_result) > 2:
            rospy.logwarn('there are clusters more than 2')
            return (np.zeros((3)), np.zeros((3)))
        # Cluster shold be devided into 2 clusters and small cluster has only 1 element
        small = 1 if len(second_result[0].indices) > len(second_result[1].indices) else 0
        large = 0 if len(second_result[0].indices) > len(second_result[1].indices) else 1
        if len(second_result[small].indices) > 1:
            rospy.logwarn('there are element in the min cluster more than 1')
            return (np.zeros((3)), np.zeros((3)))
        elif len(second_result[small].indices) == 0:
            rospy.logwarn('there is no element in small cluster')
            return (np.zeros((3)), np.zeros((3)))

        pick_target = transformed_pos[small][0]

        # sort large cluster elements by x axis on base_link coords
        sorted_poses = sorted(transformed_pos[large], key = lambda x : x[0])
        
        vectors = np.zeros((3))
        for i in range(len(sorted_poses)):
            if i == 0:
                continue
            vectors += sorted_poses[i] - sorted_poses[i-1]
        vector = vectors / (len(sorted_poses) - 1)
        goal_pos = sorted_poses[-1] + vector
        return (pick_target, goal_pos)

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
