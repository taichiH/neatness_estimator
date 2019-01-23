#!/usr/bin/env python
# coding: UTF-8
import numpy as np

import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Pose
from distance_clustering import Clustering


class ClusterBoxPublisher():
    def __init__(self):
        self.box_pub = rospy.Publisher('~output', BoundingBoxArray, queue_size=1)
        self.label_lst = rospy.get_param('~fg_class_names')
        self.thresh = rospy.get_param("~thresh", 0.2)
        rospy.Subscriber('~input', BoundingBoxArray, self.callback)

    def callback(self, box_msg):
        labeled_boxes = {}
        label_buf = []
        orientation = Pose().orientation

        for box in box_msg.boxes:
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

            for cluster in result:
                max_candidates = [boxes[i][0] + (boxes[i][1] * 0.5) for i in cluster.indices]
                min_candidates = [boxes[i][0] - (boxes[i][1] * 0.5) for i in cluster.indices]
                candidates = np.array(max_candidates + min_candidates)

                dimension = candidates.max(axis=0) - candidates.min(axis=0)
                center = candidates.min(axis=0) + (dimension * 0.5)

                tmp_box = BoundingBox()
                tmp_box.header = box_msg.header
                tmp_box.dimensions.x = dimension[0]
                tmp_box.dimensions.y = dimension[1]
                tmp_box.dimensions.z = dimension[2]
                tmp_box.pose.position.x = center[0]
                tmp_box.pose.position.y = center[1]
                tmp_box.pose.position.z = center[2]
                tmp_box.pose.orientation = orientation
                tmp_box.label = label
                bounding_box_msg.boxes.append(tmp_box)

            bounding_box_msg.header = box_msg.header
            self.box_pub.publish(bounding_box_msg)

    def get_points(self, box):
        return np.array([box.pose.position.x,
                         box.pose.position.y,
                         box.pose.position.z,
                         box.dimensions.x,
                         box.dimensions.y,
                         box.dimensions.z,]).reshape(2, 3)

if __name__=='__main__':
    rospy.init_node('cluster_box_publisher')
    cluster_box_publisher = ClusterBoxPublisher()
    rospy.spin()
