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
            '~prefix', os.path.join(os.environ['HOME'], '.ros/neatness_estimator'))

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

        print('current_dir: ', current_dir)
        print('prev_dir: ', prev_dir)
        current_color_histograms, current_geometry_histograms, current_group_neatnesses = self.get_data(current_dir)
        prev_color_histograms, prev_geometry_histograms, prev_group_neatnesses = self.get_data(prev_dir)

        # current and prev recognized items must be the same
        if len(current_color_histograms) != len(prev_color_histograms):
            rospy.logwarn('could not compare items that has different items indices')
            res.success = False
            return res

        cnt = 0
        for cur_color_hist, prev_color_hist, cur_geo_hist, prev_geo_hist in zip(
                current_color_histograms,
                prev_color_histograms,
                current_geometry_histograms,
                prev_geometry_histograms):

            index = int(cur_color_hist[0])

            cur_color_hist = map(lambda x : float(x), cur_color_hist[1:])
            prev_color_hist = map(lambda x: float(x), prev_color_hist[1:])
            cur_geo_hist = map(lambda x : float(x), cur_geo_hist[1:])
            prev_geo_hist = map(lambda x: float(x), prev_geo_hist[1:])
            cur_group_neatnesses = map(lambda x : float(x), current_group_neatnesses)
            prev_group_neatnesses = map(lambda x : float(x), prev_group_neatnesses)

            cur_color_hist = np.array(cur_color_hist)
            prev_color_hist = np.array(prev_color_hist)
            cur_color_hist = (cur_color_hist - cur_color_hist.min()) / (cur_color_hist.max() - cur_color_hist.min())
            prev_color_hist = (prev_color_hist - prev_color_hist.min()) / (prev_color_hist.max() - prev_color_hist.min())

            color_distance = 1 - dist.cosine(
                cur_color_hist, prev_color_hist)
            geo_distance = 1 - dist.cosine(
                np.array(cur_geo_hist),np.array(prev_geo_hist))

            group_distance = 1 - abs(cur_group_neatnesses[index] -\
                                     prev_group_neatnesses[index])

            print('color hist distance: ', color_distance)
            print('geometry hist distance: ', geo_distance)
            print('group_distance', group_distance)

            plt.figure(figsize=(15,15))

            plt.subplot(4,2,1)
            plt.title('current_color_histogram-' + self.label_lst[index])
            plt.xlabel('bin')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar([i for i in range(cur_color_hist.shape[0])], cur_color_hist)

            plt.subplot(4,2,2)
            plt.title('prev_color_histogram-' + self.label_lst[index])
            plt.xlabel('bin')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar([i for i in range(len(prev_color_hist))], prev_color_hist)

            plt.subplot(4,2,3)
            plt.title('current_geometry_histogram-' + self.label_lst[index])
            plt.xlabel('bin')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar([i for i in range(len(cur_geo_hist))], cur_geo_hist)

            plt.subplot(4,2,4)
            plt.title('prev_geometry_histogram-' + self.label_lst[index])
            plt.xlabel('bin')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar([i for i in range(len(prev_geo_hist))], prev_geo_hist)

            plt.subplot(4,2,5)
            plt.title('current_group_neatnesses-' + self.label_lst[index])
            plt.xlabel('bin')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar([i for i in range(len(cur_group_neatnesses))], cur_group_neatnesses)

            plt.subplot(4,2,6)
            plt.title('prev_group_neatnesses-' + self.label_lst[index])
            plt.xlabel('bin')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar([i for i in range(len(prev_group_neatnesses))], prev_group_neatnesses)

            similarities = [color_distance, geo_distance, group_distance]
            plt.subplot(4,2,7)
            plt.title('similarities-' + self.label_lst[index])
            plt.xlabel('semantics')
            plt.ylim(ymin=0.0, ymax=1.0)
            plt.bar(['color', 'geo', 'group'], similarities, width=0.1)

            savefig_name = os.path.join(
                current_dir,
                'logs',
                'histograms_' + saved_dirs[0] + '_' + saved_dirs[1] + '_' + str(cnt) + '-' + str(index) + '.png')
            print('save to :', savefig_name)
            plt.savefig(savefig_name)

            cnt += 1

        res = SetBoolResponse()
        res.success = True
        return res


if __name__=='__main__':
    rospy.init_node('compare_data_from_file')
    compare_data = CompareData()
    rospy.spin()
