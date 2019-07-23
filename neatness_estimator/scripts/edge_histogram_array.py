#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
import message_filters

from opencv_apps.msg import Line, LineArrayStamped, Point2D
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import Label, LabelArray
from neatness_estimator_msgs.msg import EdgeHistogram, EdgeHistogramArray

class EdgeHistogramArrayPublisher():

    def __init__(self):
        self.debug = rospy.get_param('~debug', False)
        self.cv_bridge = CvBridge()
        self.scale = rospy.get_param('~scale', 0.25)
        self.approximate_sync = rospy.get_param('~approximate_sync', True)

        self.check_image_callback = False
        self.debug_img_pub = rospy.Publisher(
            '~debug_output', Image, queue_size=1)
        self.edge_histograms_pub = rospy.Publisher(
            '~output', EdgeHistogramArray, queue_size=1)

        queue_size = rospy.get_param('~queue_size', 1000)
        rospy.Subscriber('~input_rgb', Image, self.image_callback)

        sub_rects = message_filters.Subscriber(
            '~input_rects', RectArray, queue_size=queue_size, buff_size=2**24)
        sub_labels = message_filters.Subscriber(
            '~input_labels', LabelArray, queue_size=queue_size, buff_size=2**24)
        sub_edge = message_filters.Subscriber(
            '~input_edge', LineArrayStamped, queue_size=queue_size, buff_size=2**24)

        self.subs = [sub_rects, sub_labels, sub_edge]
        if self.approximate_sync:
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        sync.registerCallback(self.callback)

    def check_point_in_rect(self, lt, rb, line):
        pt1 = (line.pt1.x, line.pt1.y)
        pt2 = (line.pt2.x, line.pt2.y)
        pt1_in_rect = False
        pt2_in_rect = False

        if lt[0] < pt1[0] and pt1[0] < rb[0] and \
           lt[1] < pt1[1] and pt1[1] < rb[1]:
            pt1_in_rect = True

        if lt[0] < pt2[0] and pt2[0] < rb[0] and \
           lt[1] < pt2[1] and pt2[1] < rb[1]:
            pt2_in_rect = True

        if pt1_in_rect and pt2_in_rect:
            return True
        else:
            return False

    def callback(self, rects_msg, labels_msg, edge_msg):
        rospy.loginfo('--- callback ---')
        if not self.check_image_callback:
            rospy.logwarn('waiting for image callback ... ')
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self.rgb_msg, 'bgr8')

        color = {'black_parker' : (255, 0, 0), 'flower_shirt' : (0, 255, 0)}
        histogram_array = {}
        for label, rect in zip(labels_msg.labels, rects_msg.rects):
            if not histogram_array.has_key(label.name):
                histogram_array[label.name] = []

            lt = (int(rect.x * 1 / 0.25), int(rect.y * 1 / 0.25))
            rb = (int(rect.x * 1 / 0.25) + int(rect.width * 1 / 0.25),
                  int(rect.y * 1 / 0.25) + int(rect.height * 1 / 0.25))
            cv2.rectangle(rgb_image, lt, rb, color[label.name], 3)

            for line in edge_msg.lines:
                if self.check_point_in_rect(lt, rb, line):
                    cv2.line(rgb_image,
                             (int(line.pt1.x), int(line.pt1.y)),
                             (int(line.pt2.x), int(line.pt2.y)),
                             color[label.name], 2)

                    histogram_array[label.name].append(line)

        edge_histograms = EdgeHistogramArray()
        for key in histogram_array.keys():
            histogram = EdgeHistogram()
            histogram.label = key
            histogram.lines_num = len(histogram_array[key])
            edge_histograms.histograms.append(histogram)

            print('key: %s, array length: %d' %(key, len(histogram_array[key])))

        self.edge_histograms_pub.publish(edge_histograms)
        if self.debug:
            debug_img_msg = self.cv_bridge.cv2_to_imgmsg(rgb_image, 'bgr8')
            debug_img_msg.header = self.rgb_msg.header
            self.debug_img_pub.publish(debug_img_msg)

    def image_callback(self, image_msg):
        self.check_image_callback = True
        self.rgb_msg = image_msg


if __name__=='__main__':
    rospy.init_node('edge_histogram_array')
    edge_histogram_array = EdgeHistogramArrayPublisher()
    rospy.spin()
