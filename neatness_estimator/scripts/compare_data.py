#!/usr/bin/env python

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance as dist

import rospy
from std_srvs.srv import SetBool, SetBoolResponse

class CompareData():

    def __init__(self):


        self.prefix = rospy.get_param(
            'prefix', os.path.join(os.environ['HOME'], '.ros/neatness_estimator'))

        self.label_lst = rospy.get_param('~fg_class_names')
        rospy.Service(
            '~compare', SetBool, self.service_callback)

    def get_data(self, dir_path):
        color_histograms = []
        color_histogram_path = os.path.join(dir_path, 'data/color_histograms.csv')
        with open(color_histogram_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                row.remove('')
                color_histograms.append(row)

        geometry_histograms = []
        geometry_histogram_path = os.path.join(dir_path, 'data/geometry_histograms.csv')
        with open(geometry_histogram_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                row.remove('')
                geometry_histograms.append(row)


        group_neatnesses = []
        group_neatness_path = os.path.join(dir_path, 'data/group_neatness.csv')
        with open(group_neatness_path, 'r') as f:
            csv_data = csv.reader(f)
            for row in csv_data:
                row.pop(0)
                group_neatnesses.append(row)

        return color_histograms, geometry_histograms, group_neatnesses[1]


    def service_callback(self, req):
        saved_dirs = sorted(os.listdir(self.prefix), reverse=True)
        current_dir = os.path.join(self.prefix, saved_dirs[0])
        prev_dir = os.path.join(self.prefix, saved_dirs[1])

        current_color_histograms, current_geometry_histograms, current_group_neatnesses = self.get_data(current_dir)
        prev_color_histograms, prev_geometry_histograms, prev_group_neatnesses = self.get_data(prev_dir)

        # current and prev recognized items must be the same
        if len(current_color_histograms) != len(prev_color_histograms):
            rospy.logwarn('could not compare items that has different items indices')
            res.success = False
            return res

        target_label = 'yakisoba'
        cnt = 0
        for cur_color_hist, prev_color_hist, cur_geo_hist, prev_geo_hist in zip(
                current_color_histograms,
                prev_color_histograms,
                current_geometry_histograms,
                prev_geometry_histograms):

            index = int(cur_color_hist[0])
            if not self.label_lst[index] == target_label:
                continue

            cur_color_hist = map(lambda x : float(x), cur_color_hist[1:])
            prev_color_hist = map(lambda x: float(x), prev_color_hist[1:])
            cur_geo_hist = map(lambda x : float(x), cur_geo_hist[1:])
            prev_geo_hist = map(lambda x: float(x), prev_geo_hist[1:])


            color_distance = 1 - dist.cosine(
                np.array(cur_color_hist), np.array(prev_color_hist))
            geo_distance = 1 - dist.cosine(
                np.array(cur_geo_hist),np.array(prev_geo_hist))

            group_distance = 1 - abs(float(current_group_neatnesses[index]) -\
                                     float(prev_group_neatnesses[index]))

            print('color hist distance: ', color_distance)
            print('geometry hist distance: ', geo_distance)
            print('group_distance', group_distance)

            plt.figure(figsize=(15,15))

            plt.subplot(3,2,1)
            plt.title('current_color_histogram')
            plt.xlabel('bin')
            plt.bar([i for i in range(len(cur_color_hist))], cur_color_hist)

            plt.subplot(3,2,2)
            plt.title('prev_color_histogram')
            plt.xlabel('bin')
            plt.bar([i for i in range(len(prev_color_hist))], prev_color_hist)

            plt.subplot(3,2,3)
            plt.title('current_geometry_histogram')
            plt.xlabel('bin')
            plt.bar([i for i in range(len(cur_geo_hist))], cur_geo_hist)

            plt.subplot(3,2,4)
            plt.title('prev_geometry_histogram')
            plt.xlabel('bin')
            plt.bar([i for i in range(len(prev_geo_hist))], prev_geo_hist)

            similarities = [color_distance, geo_distance, group_distance]
            plt.subplot(3,2,5)
            plt.title('similarities')
            plt.xlabel('semantics')

            plt.bar(['color', 'geo', 'group'], similarities, width=0.1)

            savefig_name = os.path.join(
                current_dir,
                'logs',
                'histograms_' + saved_dirs[0] + '_' + saved_dirs[1] + '_' + str(cnt) + '-' + str(index) + '.png')
            plt.savefig(savefig_name)

            cnt += 1

        res = SetBoolResponse()
        res.success = True
        return res


if __name__=='__main__':
    rospy.init_node('compare_data')
    compare_data = CompareData()
    rospy.spin()
