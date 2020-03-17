#!/usr/bin/env python

import os
import csv

import rospy
import rospkg
from neatness_estimator_msgs.srv import CollectData, CollectDataResponse
from neatness_estimator_msgs.msg import AppearanceDifference

class DataCollector():

    def __init__(self):
        self.model_file_path = rospy.get_param(
            '~model_path',
            os.path.join(
                rospkg.RosPack().get_path('neatness_estimator'),
                'trained_data/sample.csv'))

        rospy.loginfo('model_path: %s' %(self.model_file_path))
        rospy.Service('~collect', CollectData, self.service_callback)

    def service_callback(self, req):
        res = CollectDataResponse()

        save_data = [req.motion_label] + \
                    [req.difference.color, req.difference.geometry, req.difference.size] + \
                    [req.obj_label]
        print('save_data', save_data)

        with open(self.model_file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(save_data)

        res.success = True
        return res

if __name__=='__main__':
    rospy.init_node('data_collector')
    data_collector = DataCollector()
    rospy.spin()
