#!/usr/bin/env python

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance as dist

import rospy
from std_srvs.srv import SetBool, SetBoolResponse
from neatness_estimator_msgs.srv import GetDifference, GetDifferenceResponse

class CompareData():

    def __init__(self):


        self.prefix = rospy.get_param(
            '~prefix', os.path.join(os.environ['HOME'], '.ros/neatness_estimator'))

        self.label_lst = rospy.get_param('~fg_class_names')

        rospy.Service(
            '~compare', GetDifference, self.service_callback)

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
        res = GetDifferenceResponse()
        if len(req.features) != 2:
            res.success = false
            return res

        prev_features = req.features[0]
        curt_features = req.features[1]

        if len(prev_features.color_histogram.histograms) == \
           len(prev_features.geometry_histogram.histograms) == \
           len(prev_features.neatness.neatness):
            rospy.logerr('prev_features element size is not same')

        if len(curt_features.color_histogram.histograms) == \
           len(curt_features.geometry_histogram.histograms) == \
           len(curt_features.neatness.neatness):
            rospy.logerr('curt_features element size is not same')


        color_distances = []
        geometry_distances = []
        group_distances = []
        for i in range(len(prev_features.color_histogram.histograms)):
            index = int(curt_features.color_histogram.histograms[i].label)

            cur_color_hist = np.array(curt_features.color_histogram.histograms[i].histogram)
            prev_color_hist = np.array(prev_features.color_histogram.histograms[i].histogram)
            cur_color_hist = (cur_color_hist - cur_color_hist.min()) / (cur_color_hist.max() - cur_color_hist.min())
            prev_color_hist = (prev_color_hist - prev_color_hist.min()) / (prev_color_hist.max() - prev_color_hist.min())

            color_distance = 1 - dist.cosine(
                cur_color_hist, prev_color_hist)
            geo_distance = 1 - dist.cosine(
                np.array(curt_features.geometory_histogram.histograms[i].histogram),
                np.array(prev_features.geometory_histogram.histograms[i].histogram))

            for i, neatness in enumerate(curt_features.neatness.neatness):
                if neatness.label == index:
                    curt_target_idx = i
            for i, neatness in enumerate(prev_features.neatness.neatness):
                if neatness.label == index:
                    prev_target_idx = i

            group_distance = 1 - abs(curt_features.neatness.neatness[curt_target_idx].group_neatness -\
                                     prev_features.neatness.neatness[prev_target_idx].group_neatness)

            color_distances.append(color_distance)
            geometry_distances.append(geometry_distance)
            group_distances.append(group_distance)

            print('---')
            print('color hist distance: ', color_distance)
            print('geometry hist distance: ', geo_distance)
            print('group_distance', group_distance)

        res = GetDifferenceResponse()
        res.color_distance = color_distances
        res.geometry_distance = geometry_distances
        res.group_distance = group_distances
        res.success = True
        return res

if __name__=='__main__':
    rospy.init_node('compare_data')
    compare_data = CompareData()
    rospy.spin()
