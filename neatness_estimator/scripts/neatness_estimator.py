#!/usr/bin/env python
# coding: UTF-8
import os
import datetime
import pandas as pd
import cv2
import cv_bridge
import numpy as np
import rospy
import rospkg
import message_filters
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from jsk_recognition_msgs.msg import ClassificationResult
from opencv_apps.msg import LineArrayStamped, Point2D
from cv_bridge import CvBridge, CvBridgeError
from neatness_estimator.msg import Neatness

class NeatnessWatcher():

    def __init__(self):
        self.label_lst = ["alfort", "alfortwhite", "almond", "apc_shelf_flont",\
                           "coffee", "consome", "dars", "darsmilk", "darswhite",\
                           "donbe", "kinoko", "macadamia", "milk", "mixjuice",\
                           "marble", "norishio", "pie", "shelf_flont", "takenoko",\
                           "tee", "xylitop", "yakisoba"]

        self.neatness_pub = rospy.Publisher('~output',
                                           Neatness,
                                           queue_size=1)
        self.group_dist_array = {}
        self.filling_dist_array = {}
        self.pulling_dist_array = {}

        self.output_data = {'neatness':[], 'group_neatness':[],
                            'filling_neatness':[], 'pulling_neatness':[]}
        self.output_dir = os.path.join(rospkg.RosPack().get_path('neatness_estimator'), 'output')

        self.save_log =rospy.get_param('~save_log', False)
        self.boxes = []
        self.bridge = CvBridge()
        self.subscribe()
        pass

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)

        sub_cluster_box = message_filters.Subscriber(
            '~input/cluster_boxes',
            BoundingBoxArray,queue_size=1, buff_size=2**24)
        sub_instance_box = message_filters.Subscriber(
            '~input/instance_boxes',
            BoundingBoxArray, queue_size=1, buff_size=2**24)
        # sub_line = message_filters.Subscriber(
        #     '~input_line', LineArrayStamped, queue_size=1, buff_size=2**24)

        self.subs = [sub_instance_box, sub_cluster_box]

        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self.callback)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def callback(self, instance_msg, cluster_msg):
        print('callback')
        bridge = self.bridge
        labeled_boxes = {}
        category_boxes = {}
        label_buf = []

        for box in cluster_msg.boxes:
            category_boxes[box.label] = self.get_array(box)

        for box in instance_msg.boxes:
            if box.label in label_buf:
                labeled_boxes[box.label] += [self.get_array(box)]
            else:
                labeled_boxes[box.label] = [self.get_array(box)]
                label_buf.append(box.label)

        group_dist = self.calc_group_dist(category_boxes, labeled_boxes, label_buf)
        group_dist_mean = np.array(group_dist.values()).mean()

        filling_dist = self.calc_filling_dist(category_boxes, label_buf)
        filling_dist_mean = 1 - np.array(filling_dist.values()).mean()

        pulling_dist = self.calc_pulling_dist(category_boxes, label_buf)
        pulling_dist_mean = 1 - np.array(pulling_dist.values()).mean()

        neatness = np.array([group_dist_mean, filling_dist_mean, pulling_dist_mean]).mean()

        neatness_msg = Neatness()
        neatness_msg.header = instance_msg.header
        neatness_msg.group_neatness = group_dist_mean
        neatness_msg.filling_neatness = filling_dist_mean
        neatness_msg.pulling_neatness = pulling_dist_mean
        neatness_msg.neatness = neatness
        self.neatness_pub.publish(neatness_msg)

        if self.save_log:
            self.output_data['neatness'].append(neatness)
            self.output_data['group_neatness'].append(group_dist_mean)
            self.output_data['filling_neatness'].append(filling_dist_mean)
            self.output_data['pulling_neatness'].append(pulling_dist_mean)
            # file_name = 'neatness_{0:%Y%m%d-%H%M%S}.csv'.format(datetime.datetime.now())
            neatness_log = 'neatness_output.csv'
            output_file = os.path.join(self.output_dir, neatness_log)
            pd.DataFrame(data=self.output_data).to_csv(output_file)

            for key, val in zip(group_dist.keys(), group_dist.values()):
                if self.group_dist_array.has_key(key):
                    self.group_dist_array[key] += [val]
                else:
                    self.group_dist_array[key] = [val]

            for key, val in zip(filling_dist.keys(), filling_dist.values()):
                if self.filling_dist_array.has_key(key):
                    self.filling_dist_array[key] += [val]
                else:
                    self.filling_dist_array[key] = [val]

            for key, val in zip(pulling_dist.keys(), pulling_dist.values()):
                if self.pulling_dist_array.has_key(key):
                    self.pulling_dist_array[key] += [val]
                else:
                    self.pulling_dist_array[key] = [val]

            items_group_neatness = 'items_group_neatness.csv'
            items_group_neatness_output = os.path.join(self.output_dir, items_group_neatness)
            pd.DataFrame(data=self.group_dist_array).to_csv(items_group_neatness_output)

            items_filling_neatness = 'items_filling_neatness.csv'
            items_filling_neatness_output = os.path.join(self.output_dir, items_filling_neatness)
            pd.DataFrame(data=self.filling_dist_array).to_csv(items_filling_neatness_output)

            items_pulling_neatness = 'items_pulling_neatness.csv'
            items_pulling_neatness_output = os.path.join(self.output_dir, items_pulling_neatness)
            pd.DataFrame(data=self.pulling_dist_array).to_csv(items_pulling_neatness_output)

        print('neatness, group_dist_mean, filling_dist_mean, pulling_dist_mean')
        print(neatness, group_dist_mean, filling_dist_mean, pulling_dist_mean)

    def get_array(self, box):
        array = np.array([box.pose.position.x,
                          box.pose.position.y,
                          box.pose.position.z,
                          box.dimensions.x,
                          box.dimensions.y,
                          box.dimensions.z])
        return array.reshape(2, 3)

    def calc_group_dist(self, category_boxes, labeled_boxes, labels):
        group_dist = {}
        for label in labels:
            if not label == self.label_lst.index('shelf_flont'):
                item_vol = 0
                for item in labeled_boxes[label]:
                    item_vol += item.prod(1)[1]
                category_vol = category_boxes[label].prod(1)[1]
                group_dist[label] = item_vol / category_vol
        return group_dist

    def calc_filling_dist(self, category_boxes, labels):
        filling_dist = {}
        shelf_i = self.label_lst.index('shelf_flont')
        shelf_lt = np.array(np.array([category_boxes[shelf_i][0][0] + category_boxes[shelf_i][1][0]* 0.5,
                                      category_boxes[shelf_i][0][1] + category_boxes[shelf_i][1][1]* 0.5,
                                      category_boxes[shelf_i][0][2] + category_boxes[shelf_i][1][2]* 0.5]))
        shelf_rb = np.array(np.array([category_boxes[shelf_i][0][0] - category_boxes[shelf_i][1][0]* 0.5,
                                      category_boxes[shelf_i][0][1] - category_boxes[shelf_i][1][1]* 0.5,
                                      category_boxes[shelf_i][0][2] - category_boxes[shelf_i][1][2]* 0.5]))

        l_shelf = category_boxes[self.label_lst.index('shelf_flont')][1][2]

        # sorted by position y value
        sorted_boxes = sorted(category_boxes.items(), key = lambda x : x[1][0][1])

        offset = 0.15
        for i in range(1, len(sorted_boxes)):
            if sorted_boxes[i][1][0][1] < (shelf_lt[1] + offset) and \
               sorted_boxes[i][1][0][1] > (shelf_rb[1] - offset) and \
               sorted_boxes[i][1][0][2] > (shelf_lt[2] - offset) and \
               not i == shelf_i:
                key = str(i) + '-' + str(i-1)
                filling_dist[key] = np.linalg.norm(sorted_boxes[i][1][0] - sorted_boxes[i-1][1][0]) / l_shelf

        return filling_dist

    def calc_pulling_dist(self, category_boxes, labels):
        pulling_dist = {}
        # shelf_front = box_center.x - box_dimension.x * 0.5
        shelf_front = category_boxes[self.label_lst.index('shelf_flont')][0][0] -\
                      category_boxes[self.label_lst.index('shelf_flont')][1][0] * 0.5

        # TODO estimate l_depth
        l_depth = 0.4

        for i in category_boxes.keys():
            if not i == self.label_lst.index('shelf_flont'):
                item_front = category_boxes[i][0][0] - category_boxes[i][1][0] * 0.5
                pulling_dist[i] = (item_front - shelf_front) / l_depth

        return pulling_dist

if __name__ == '__main__':
    rospy.init_node('neatness_watcher')
    neatness_watcher = NeatnessWatcher()
    rospy.spin()
