#!/usr/bin/env python

import rospy
from neatness_estimator_msgs.srv import GetDifference, GetDifferenceResponse
from std_msgs.msg import Float32

class DiffWatcher:

    def __init__(self):
        rospy.wait_for_service(
            '/estimation_module_interface_color_and_geometry/call')

        # waiting for  3 sec for msg callback
        rospy.sleep(3.)

        self.client = rospy.ServiceProxy(
            '/estimation_module_interface_color_and_geometry/call',
            GetDifference)
        self.color_pub = rospy.Publisher('~color_distance', Float32, queue_size=1)
        self.geo_pub = rospy.Publisher('~geometry_distance', Float32, queue_size=1)

        rate = rospy.get_param('~rate', 1)
        rospy.Timer(rospy.Duration(1. / rate), self.callback)

    def callback(self, event):
        res = self.client(task='two_scene')

        if len(res.color_distance) == 0 or len(res.geometry_distance) == 0:
            return

        color_dist = res.color_distance[0]
        geo_dist = res.geometry_distance[0]
        print(color_dist, geo_dist)

        color_dist_msg = Float32(data=color_dist)
        geo_dist_msg = Float32(data=geo_dist)
        self.color_pub.publish(color_dist_msg)
        self.geo_pub.publish(geo_dist_msg)



if __name__=='__main__':
    rospy.init_node('difference_estimation_client')
    diff_watcher = DiffWatcher()
    rospy.spin()
