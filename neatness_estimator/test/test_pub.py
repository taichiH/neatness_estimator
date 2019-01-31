#!/usr/bin/env python
# coding: UTF-8

import rospy
import numpy as np
from neatness_estimator_msgs.msg import TargetAndGoal, TargetAndGoalArray
from geometry_msgs.msg import Point

rospy.init_node('test_pub')
test_pub = rospy.Publisher('/test_pub/target_and_goal_array', TargetAndGoalArray, queue_size=1)
test_array = np.random.rand(3,2,3)

while not rospy.is_shutdown():
    target_and_goals = TargetAndGoalArray()
    for a in test_array:
        target_and_goal = TargetAndGoal()
        target_and_goal.target.point = Point(a[0][0], a[0][1], a[0][2])
        target_and_goal.goal.point = Point(a[1][0], a[1][1], a[1][2])
        target_and_goals.positions.append(target_and_goal)
        
    test_pub.publish(target_and_goals)
    rospy.sleep(1)
