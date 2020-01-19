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
        method = rospy.get_param('~compare_method', 'bray')
        if method == 'bray':
            self.compare_method = dist.braycurtis
        elif method == 'cosine':
            self.compare_method = dist.cosine
        else:
            rospy.logerr('set compare method from [cosine, bray]')

        self.correct_feature_path = rospy.get_param(
            '~correct_feature_path',
            os.path.join(os.environ['HOME'], '.ros/correct_feature'))
        self.correct_color_feature = {0 : []}
        self.correct_geometry_feature = {0 : []}
        self.get_correct_feature()
        self.pick_target

        rospy.Service(
            '~estimate', GetDifference, self.service_callback)

    def get_correct_feature(self):
        color_feature_path = os.path.join(self.correct_feature_path, 'correct_color_feature.csv')
        geometry_feature_path = os.path.join(self.correct_feature_path, 'correct_geometry_feature.csv')

        with open(color_feature_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                item_label = int(row[0])
                feature = (map(lambda x : float(x), row[1:len(row[1:])]
                self.correct_color_feature[item_label] = feature

        with open(geometry_feature_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                item_label = int(row[0])
                feature = (map(lambda x : float(x), row[1:]))
                self.correct_geometry_feature[item_label] = feature

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
            index = int(curt_features.color_histogram.histograms[i].label) # label number

            cur_color_hist = np.array(curt_features.color_histogram.histograms[i].histogram)
            prev_color_hist = np.array(prev_features.color_histogram.histograms[i].histogram)
            cur_color_hist = (cur_color_hist - cur_color_hist.min()) / (cur_color_hist.max() - cur_color_hist.min())
            prev_color_hist = (prev_color_hist - prev_color_hist.min()) / (prev_color_hist.max() - prev_color_hist.min())

            color_distance = 1 - self.compare_method(
                cur_color_hist, prev_color_hist)
            geometry_distance = 1 - self.compare_method(
                np.array(curt_features.geometry_histogram.histograms[i].histogram),
                np.array(prev_features.geometry_histogram.histograms[i].histogram))

            cur_color_distance = 1 - self.compare_method(
                cur_color_hist, self.correct_color_feature[index])
            prev_color_distance = 1 - self.compare_method(
                prev_color_hist, self.correct_color_feature[index])
            cur_geometry_distance = 1 - self.compare_method(
                curt_features.geometry_histogram.histograms[i].histogram,
                self.correct_geometry_feature[index])
            prev_geometry_distance = 1 - self.compare_method(
                prev_features.geometry_histogram.histograms[i].histogram,
                self.correct_geometry_feature[index])

            cur_distance_ave = (cur_color_distance + cur_geometry_distance) / 2
            prev_distance_ave = (prev_color_distance + prev_geometry_distance) / 2
            if cur_distance_ave < prev_distance_ave:
                self.pick_target = 'prev'
            else:
                self.pick_target = 'cur'

            self.correct_geometry_feature
            ##

            curt_target_idx = 0
            prev_target_idx = 0
            for i, neatness in enumerate(curt_features.neatness.neatness):
                if neatness.label == index:
                    curt_target_idx = i
            for i, neatness in enumerate(prev_features.neatness.neatness):
                if neatness.label == index:
                    prev_target_idx = i

            # difference between tow scene
            # group_distance = 1 - abs(
            #     curt_features.neatness.neatness[curt_target_idx].group_neatness -\
            #     prev_features.neatness.neatness[prev_target_idx].group_neatness)

            # difference between two items
            # group_distance = curt_features.neatness.neatness[curt_target_idx].group_neatness

            # calc volume ratio
            curt_volume = curt_features.neatness.neatness[0].group_neatness
            prev_volume = prev_features.neatness.neatness[0].group_neatness
            if curt_volume > prev_volume:
                group_distance = prev_volume / curt_volume
            else:
                group_distance = curt_volume / prev_volume

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
        res.message = self.pick_target
        res.success = True
        return res

if __name__=='__main__':
    rospy.init_node('distance_estimator')
    distance_estimator = DistanceEstimator()
    rospy.spin()
