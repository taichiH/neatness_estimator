#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy

import message_filters
from jsk_recognition_msgs.msg import Rect, RectArray
from jsk_recognition_msgs.msg import ClassificationResult
from what_i_see_msgs.msg import Scored2DBox, Scored2DBoxArray

class Rect2LabeledArray():

    def __init__(self):
        self.pub = rospy.Publisher('~output/boxes', Scored2DBoxArray, queue_size=1)
        self.subscribe()

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 100)

        sub_box = message_filters.Subscriber(
            '~input/rect', RectArray, queue_size=queue_size)
        sub_class = message_filters.Subscriber(
            '~input/class', ClassificationResult, queue_size=queue_size)

        self.subs = [sub_box, sub_class]
        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.convertCb)

    def convertCb(self, boxes_msg, classes_msg):
        boxes = Scored2DBoxArray()
        boxes.header = boxes_msg.header
        for rect, label, proba in zip(boxes_msg.rects,
                                      classes_msg.label_names,
                                      classes_msg.label_proba):
            box = Scored2DBox()
            box.x = rect.x
            box.y = rect.y
            box.width = rect.width
            box.height = rect.height
            box.label = label
            box.score = proba
            boxes.boxes.append(box)
        self.pub.publish(boxes)

if __name__ == '__main__':
    rospy.init_node("rect_to_labeledarray")
    r2la = Rect2LabeledArray()
    rospy.spin()
