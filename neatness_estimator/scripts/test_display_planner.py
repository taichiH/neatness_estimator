#!/usr/bin/env python

import rospy
from neatness_estimator_msgs.srv import DisplayState

rospy.init_node('test_display_planner')
display_client = rospy.ServiceProxy('/display_planner_server', DisplayState)
res = display_client(reference_state=[12,12,12,12,4,4,4,4,16,16,16,16], border_indexes=[0,4,8,12])

print(res.status)
print('-----------------------')
print(res.plan)
print('-----------------------')
print(res.distance)
