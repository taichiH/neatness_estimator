#!/usr/bin/env python
# coding: UTF-8
import os
import numpy as np
import rospy
import rospkg
import message_filters
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_recognition_msgs.msg import ClassificationResult
from neatness_estimator_msgs.msg import Neatness
from geometry_msgs.msg import Pose
from distance_clustering import Clustering

class NeatnessEstimator():

    def __init__(self):
        self.label_lst = rospy.get_param('~fg_class_names')

        self.box_pub = rospy.Publisher('~output_box',
                                       BoundingBoxArray,
                                       queue_size=1)
        self.neatness_pub = rospy.Publisher('~output_neatness',
                                           Neatness,
                                           queue_size=1)

        self.thresh = rospy.get_param('~thresh', 0.8)
        self.boxes = []
        self.subscribe()
        pass

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
        for label, boxes in zip(labeled_boxes.keys(), labeled_boxes.values()):
            thresh = self.thresh
            if self.label_lst[label] == 'shelf_flont':
                thresh = 2.0

            clustering = Clustering()
            boxes = np.array(boxes)
            result = clustering.clustering_wrapper(boxes, thresh)

            for i, cluster  in enumerate(result):                
                distances = self.get_distances([boxes[i][0] for i in cluster.indices])
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

        bounding_box_msg.header = instance_msg.header
        self.box_pub.publish(bounding_box_msg)

    def get_distances(self, centers):
        distances = []
        for i, center_a in enumerate(centers):
            for j, center_b in enumerate(centers):
                if i == j:
                    continue
                norm = np.linalg.norm(center_a - center_b)
                distances.append(norm)
        return np.array(list(set(distances)))

    def get_points(self, box):
        return (np.array([box.pose.position.x,
                          box.pose.position.y,
                          box.pose.position.z,
                          box.dimensions.x,
                          box.dimensions.y,
                          box.dimensions.z])).reshape(2, 3)

if __name__ == '__main__':
    rospy.init_node('neatness_estimator')
    neatness_estimator = NeatnessEstimator()
    rospy.spin()
