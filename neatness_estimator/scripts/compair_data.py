#!/usr/bin/env python

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_srvs.srv import SetBool, SetBoolResponse

class CompairData():

    def __init__(self):


        self.prefix = rospy.get_param(
            'prefix', os.path.join(os.environ['HOME'], '.ros/neatness_estimator'))

        self.label_lst = rospy.get_param('~fg_class_names')
        rospy.Service(
            '~compair', SetBool, self.service_callback)

    def get_data(self, dir_path):
        group_neatness_path = os.path.join(dir_path, 'data/group_neatness.csv')
        with open(group_neatness_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                # print(row)
                pass

        color_histograms = []
        color_histogram_path = os.path.join(dir_path, 'data/color_histograms.csv')
        with open(color_histogram_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                color_histograms.append(row)

        geometry_histogram_path = os.path.join(dir_path, 'data/geometry_histograms.csv')
        with open(geometry_histogram_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                # print(row)
                pass

        return color_histograms

    def service_callback(self, req):
        saved_dirs = sorted(os.listdir(self.prefix), reverse=True)
        current_dir = os.path.join(self.prefix, saved_dirs[0])
        prev_dir = os.path.join(self.prefix, saved_dirs[1])

        current_color_histograms = self.get_data(current_dir)
        prev_color_histograms = self.get_data(prev_dir)

        # current and prev recognized items must be the same
        if len(current_color_histograms) != len(prev_color_histograms):
            rospy.logwarn('could not compare items that has different items indices')
            res.success = False
            return res

        for current_histogram, prev_histogram in zip(
                current_color_histograms, prev_color_histograms):
            index = int(current_histogram[0])
            if not self.label_lst[index] == 'yakisoba':
                continue

            print(self.label_lst[index], current_histogram[1:])
            current_histogram.remove('')
            sample_current_hist = map(lambda x : float(x), current_histogram[1:])

            index = int(prev_histogram[0])
            print(self.label_lst[index], prev_histogram[1:])
            sample_prev_hist = np.array(prev_histogram[1:])

        x_axes = [i for i in range(len(sample_current_hist))]
        print(x_axes)
        print(sample_current_hist)

        plt.figure()
        plt.subplot(1,1,1)
        plt.xlabel('bin')
        plt.title('sample_current_hist')
        plt.bar(x_axes, sample_current_hist)
        plt.show()

        res = SetBoolResponse()
        res.success = True
        return res


if __name__=='__main__':
    rospy.init_node('compair_data')
    compair_data = CompairData()
    rospy.spin()
