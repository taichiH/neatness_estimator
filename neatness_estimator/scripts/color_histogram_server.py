#!/usr/bin/env python
# coding: UTF-8

import numpy as np
import cv2

import rospy
from neatness_estimator_msgs.srv import GetColorHistogram, GetColorHistogramResponse
from cv_bridge import CvBridge, CvBridgeError


class ColorHistogramServer:

    def __init__(self):
        self.cv_bridge = CvBridge()
        rospy.Service(
            '~get_color_histogram', GetColorHistogram, self.service_callback)

    def get_histogram(self, image, mask):
        color = ('b','g','r')
        hists = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], mask, [256], [0,256])
            hist = hist.T
            hists.append(hist[0])

        hists = np.array(hists).T
        vec = np.zeros((hists.shape[0]), dtype=np.float64)
        for i in range(hists.shape[0]):
            vec[i] = np.linalg.norm(hists[i])

        return vec

    def service_callback(self, req):
        image_msg = req.image
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        mask_msg = req.mask
        mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, 'mono8')

        if image.shape[0] * image.shape[1] == 0:
            rospy.logwarn('input image size: %d', image.shape[0] * image.shape[1])
            res.success = False
            return res

        hist = self.get_histogram(image, mask)
        hist = hist / hist.max()

        res = GetColorHistogramResponse()
        res.histogram.histogram = list(hist)
        res.success = True
        return res

if __name__ == "__main__":
    rospy.init_node("color_histogram_server")
    color_histogram_server = ColorHistogramServer()
    rospy.spin()
