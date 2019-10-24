#!/usr/bin/env python
# coding: UTF-8

import numpy as np
import cv2

import rospy
from jsk_recognition_msgs.msg import ColorHistogram
from neatness_estimator_msgs.srv import GetColorHistogram, GetColorHistogramResponse
from cv_bridge import CvBridge, CvBridgeError


class ColorHistogramServer:

    def __init__(self):
        self.cv_bridge = CvBridge()
        rospy.Service(
            '~get_color_histogram', GetColorHistogram, self.service_callback)

    def service_callback(self, req):
        image_msg = req.image
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        mask_msg = req.mask
        mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, 'mono8')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = np.array(cv2.calcHist([image], [0], mask, [256], [0,256]))
        hist = hist.reshape(1,256)[0]

        res = GetColorHistogramResponse()
        res.histogram.histogram = list(hist)
        return res

if __name__ == "__main__":
    rospy.init_node("color_histogram_server")
    color_histogram_server = ColorHistogramServer()
    rospy.spin()
