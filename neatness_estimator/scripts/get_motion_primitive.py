#!/usr/bin/env python

import rospy
from neatness_estimator_msgs.srv import GetMotionPrimitive, GetMotionPrimitiveResponse

class GetMotionPrimitiveServer():

    def __init__(self):
        rospy.Service('~', GetMotionPrimitive, self.service_callback)

    def run(self, color, geometry, group):
        # random forest

        motion_primitive = 'rotate'
        return motion_primitive

    def service_callback(self, req):
        res = GetMotionPrimitiveResponse()

        motion_primitives = []
        for color, geo, group in zip(
                req.color_distance,
                req.geometry_distance,
                req.group_distance):
            motion_primitive = run(color, geo, group)
            motion_primitives.append(motion_primitive)

        res.motions = motion_primitives
        res.success = True
        return res

if __name__=='__main__':
    rospy.init_node('get_motion_primitive_service')
    get_motion_primitive_server = GetMotionPrimitiveServer()
    rospy.spin()
