#!/usr/bin/env python

import os
import csv

import rospy
from std_srvs.srv import SetBool, SetBoolResponse

class CompairData():

    def __init__(self):
        home_dir = os.environ['HOME']
        self.prefix = rospy.get_param(
            'prefix', os.path.join(home_dir, '.ros/neatness_estimator'))

        rospy.Service(
            '~compair', SetBool, self.service_callback)

    def service_callback(self, req):
        saved_dirs = sorted(os.listdir(self.prefix), reverse=True)
        current_dir = os.path.join(self.prefix, saved_dirs[0])
        prev_dir = os.path.join(self.prefix, saved_dirs[1])

        group_neatness_path = os.path.join(current_dir, 'data/group_neatness.csv')
        print(group_neatness_path)
        with open(group_neatness_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                print(row)

        color_histogram_path = os.path.join(current_dir, 'data/color_histograms.csv')
        print(color_histogram_path)
        with open(color_histogram_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                print(row)

        geometry_histogram_path = os.path.join(current_dir, 'data/geometry_histograms.csv')
        print(geometry_histogram_path)
        with open(geometry_histogram_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                print(row)


if __name__=='__main__':
    rospy.init_node('compair_data')
    compair_data = CompairData()
    rospy.spin()
