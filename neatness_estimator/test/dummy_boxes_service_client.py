#!/usr/bin/env python

import rospy
from neatness_estimator_msgs.srv import UpdateBoundingBox
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

class DummyBoxesServiceClient():

    def __init__(self):
        rospy.Subscriber(
            '/dummy_bounding_box_publisher/output', BoundingBoxArray, self.callback)

        self. client = rospy.ServiceProxy(
            '/dummy_bounding_box_publisher/update_boxes', UpdateBoundingBox)

    def callback(self, boxes_msg):
        self.client(boxes = boxes_msg)


if __name__=='__main__':
    rospy.init_node('dummy_boxes_service_client')
    dbsc = DummyBoxesServiceClient()
    rospy.spin()
