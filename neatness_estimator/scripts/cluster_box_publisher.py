#!/usr/bin/env python
# coding: UTF-8
import numpy as np

import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from distance_clustering import Clustering

class ClusterBoxPublisher():
    def __init__(self):
        self.box_pub = rospy.Publisher('~output', BoundingBoxArray, queue_size=1)
        rospy.Subscriber('~input', BoundingBoxArray, self.callback)

    def callback(box_msg):
        labeled_boxes = {}
        label_buf = []

        for box in box_msg.boxes:
            if box.label in label_buf:
                labeled_boxes[box.label] += [self.get_points(box)]
            else:
                labeled_boxes[box.label] = [self.get_points(box)]
                label_buf.append(box.label)
            
        for label, boxes in zip(labeled_boxes.keys(), labeled_boxes.values()):
            print(label, boxes)

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
