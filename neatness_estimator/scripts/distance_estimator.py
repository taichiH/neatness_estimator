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

class DistanceEstimator():

    def __init__(self):
        self.prefix = rospy.get_param(
            '~prefix', os.path.join(os.environ['HOME'], '.ros/neatness_estimator'))

        self.label_lst = rospy.get_param('~fg_class_names')

        rospy.Service(
            '~estimate', GetDifference, self.service_callback)

    def service_callback(self, req):
        rospy.loginfo('distance_estimator service called')

        res = GetDifferenceResponse()
        if len(req.features) != 2:
            rospy.loginfo('requested features size: %d', len(req.features))
            res.success = False
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


        labels = []
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
            geometry_distance = 1 - dist.cosine(
                np.array(curt_features.geometry_histogram.histograms[i].histogram),
                np.array(prev_features.geometry_histogram.histograms[i].histogram))

            curt_target_idx = 0
            prev_target_idx = 0
            for i, neatness in enumerate(curt_features.neatness.neatness):
                if neatness.label == index:
                    curt_target_idx = i
            for i, neatness in enumerate(prev_features.neatness.neatness):
                if neatness.label == index:
                    prev_target_idx = i

            group_distance = 1 - abs(curt_features.neatness.neatness[curt_target_idx].group_neatness -\
                                     prev_features.neatness.neatness[prev_target_idx].group_neatness)

            labels.append(index)
            color_distances.append(color_distance)
            geometry_distances.append(geometry_distance)
            group_distances.append(group_distance)

            print('---')
            print('label: ', self.label_lst[index])
            print('color hist distance: ', color_distance)
            print('geometry hist distance: ', geometry_distance)
            print('group_distance', group_distance)

        res = GetDifferenceResponse()
        res.labels = labels
        res.color_distance = color_distances
        res.geometry_distance = geometry_distances
        res.group_distance = group_distances
        res.success = True
        return res

if __name__=='__main__':
    rospy.init_node('distance_estimator')
    distance_estimator = DistanceEstimator()
    rospy.spin()
