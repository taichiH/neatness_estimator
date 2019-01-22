#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy
import message_filters

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_recognition_msgs.msg import LabelArray

class LabeledBoundingBoxPublisher():

    def __init__(self):
        self.label_list = ["alfort", "alfortwhite", "almond", "apc_shelf_flont",\
                           "coffee", "consome", "dars", "darsmilk", "darswhite",\
                           "donbe", "kinoko", "macadamia", "milk", "mixjuice",\
                           "marble", "norishio", "pie", "shelf_flont", "takenoko",\
                           "tee", "xylitop", "yakisoba", "hand"]

        self.with_cluster_box = rospy.get_param('~with_cluster_box', False)
        self.labeled_cluster_boxes_pub = rospy.Publisher('~output/labeled_cluster_boxes',
                                                         BoundingBoxArray,
                                                         queue_size=1)
        self.labeled_instance_boxes_pub = rospy.Publisher('~output/labeled_instance_boxes',
                                                          BoundingBoxArray,
                                                          queue_size=1)
        self.subscribe()

    def subscribe(self):
        if self.with_cluster_box:
            queue_size = rospy.get_param('~queue_size', 100)
            sub_cluster_boxes = message_filters.Subscriber(
                '~input/cluster_boxes', BoundingBoxArray, queue_size=queue_size)
            sub_instance_boxes = message_filters.Subscriber(
                '~input/instance_boxes', BoundingBoxArray, queue_size=queue_size)
            sub_instance_labels = message_filters.Subscriber(
                '~input/instance_labels', LabelArray, queue_size=queue_size)

            self.subs = [sub_cluster_boxes, sub_instance_boxes, sub_instance_labels]
            if rospy.get_param('~approximate_sync', False):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            sync.registerCallback(self.callback_with_cluster_box)
        else:
            queue_size = rospy.get_param('~queue_size', 100)
            sub_instance_boxes = message_filters.Subscriber(
                '~input/instance_boxes', BoundingBoxArray, queue_size=queue_size)
            sub_instance_labels = message_filters.Subscriber(
                '~input/instance_labels', LabelArray, queue_size=queue_size)

            self.subs = [sub_instance_boxes, sub_instance_labels]
            if rospy.get_param('~approximate_sync', False):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            sync.registerCallback(self.callback)

    def callback_with_cluster_box(self, cluster_boxes_msg, instance_boxes_msg, instance_label_msg):
        labeled_cluster_boxes = BoundingBoxArray()
        labeled_instance_boxes = BoundingBoxArray()

        labeled_cluster_boxes.header = cluster_boxes_msg.header
        labeled_instance_boxes.header = instance_boxes_msg.header

        for index, box in enumerate(cluster_boxes_msg.boxes):
            if not box.pose.position.x == 0.0:
                tmp_box = BoundingBox()
                tmp_box.header = box.header
                tmp_box.pose = box.pose
                tmp_box.dimensions = box.dimensions

                # TODO fix index indent, jsk_pcl_ros_utils/label_to_cluster_point_indices_nodelet.cpp
                tmp_box.label = index + 1

                labeled_cluster_boxes.boxes.append(tmp_box)

        for box, label in zip(instance_boxes_msg.boxes, instance_label_msg.labels):
            tmp_box = BoundingBox()
            tmp_box.header = box.header
            tmp_box.pose = box.pose
            tmp_box.dimensions = box.dimensions
            tmp_box.label = label.id
            labeled_instance_boxes.boxes.append(tmp_box)

        self.labeled_cluster_boxes_pub.publish(labeled_cluster_boxes)
        self.labeled_instance_boxes_pub.publish(labeled_instance_boxes)

    def callback(self, instance_boxes_msg, instance_label_msg):
        labeled_instance_boxes = BoundingBoxArray()
        labeled_instance_boxes.header = instance_boxes_msg.header

        for box, label in zip(instance_boxes_msg.boxes, instance_label_msg.labels):
            tmp_box = BoundingBox()
            tmp_box.header = box.header
            tmp_box.pose = box.pose
            tmp_box.dimensions = box.dimensions
            tmp_box.label = label.id
            labeled_instance_boxes.boxes.append(tmp_box)

        self.labeled_instance_boxes_pub.publish(labeled_instance_boxes)

if __name__ == '__main__':
    rospy.init_node("labeled_bounding_box_publisher")
    lbbp = LabeledBoundingBoxPublisher()
    rospy.spin()
