#!/usr/bin/env python

import rospy
import numpy as np
import message_filters
import copy

from threading import Lock

from jsk_recognition_msgs.msg import ClusterPointIndices
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import LabelArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image

from neatness_estimator_msgs.srv import GetFeatures, GetFeaturesResponse, GetFeaturesRequest
from std_srvs.srv import SetBool, SetBoolResponse

# /openni_camera/point_cloud/cloud_registered/remote
# /openni_camera/openni_camera/rgb/image_rect_color
# /multi_euclidean_clustering/output_indices
# /mask_rcnn_instance_segmentation/output/labels
# /labeled_bounding_box_publisher_aligned/output/labeled_instance_boxes
# /labeled_bounding_box_publisher_aligned/output/labeled_cluster_boxes

class Sample():

    def __init__(self):
        self.lock = Lock()

        self.cloud_msg = PointCloud2()
        self.image_msg = Image()
        self.cluster_msg = ClusterPointIndices()
        self.labels_msg = LabelArray()
        self.instance_boxes_msg = BoundingBoxArray()
        self.cluster_boxes_msg = BoundingBoxArray()

        rospy.Service(
            '~call', SetBool, self.service_callback)

        self.get_feature_client = rospy.ServiceProxy(
            '/objects_feature_extractor/extract', GetFeatures)

        sub_cloud = message_filters.Subscriber(
            '/openni_camera/point_cloud/cloud_registered/remote', PointCloud2, queue_size=100)
        sub_image = message_filters.Subscriber(
            '/openni_camera/rgb/image_rect_color', Image, queue_size=100)
        sub_cluster = message_filters.Subscriber(
            '/multi_euclidean_clustering/output_indices', ClusterPointIndices, queue_size=100)
        sub_labels = message_filters.Subscriber(
            '/mask_rcnn_instance_segmentation/output/labels', LabelArray, queue_size=100)
        sub_instance_boxes = message_filters.Subscriber(
            '/labeled_bounding_box_publisher_aligned/output/labeled_instance_boxes', BoundingBoxArray, queue_size=100)
        sub_cluster_boxes = message_filters.Subscriber(
            '/labeled_bounding_box_publisher_aligned/output/labeled_cluster_boxes', BoundingBoxArray, queue_size=100)

        self.subs = [sub_cloud, sub_image, sub_cluster, sub_labels, sub_instance_boxes, sub_cluster_boxes]
        sync = message_filters.ApproximateTimeSynchronizer(
            fs=self.subs, queue_size=100, slop=0.1)
        sync.registerCallback(self.callback)

    def callback(
            self, cloud_msg, image_msg, cluster_msg, labels_msg, instance_boxes_msg, cluster_boxes_msg):
        with self.lock:
            self.cloud_msg = cloud_msg
            self.image_msg = image_msg
            self.cluster_msg = cluster_msg
            self.labels_msg = labels_msg
            self.instance_boxes_msg = instance_boxes_msg
            self.cluster_boxes_msg = cluster_boxes_msg
            print('------------------------------------- callback')

    def call(self, index):
        print('---- index: ', index)

        print('len(self.instance_boxes_msg): ', len(self.instance_boxes_msg.boxes))

        instance_boxes_msg = copy.deepcopy(self.instance_boxes_msg)
        del instance_boxes_msg.boxes[:]
        instance_boxes_msg.boxes.append(self.instance_boxes_msg.boxes[index])
        cluster_boxes_msg = copy.deepcopy(self.cluster_boxes_msg)
        del cluster_boxes_msg.boxes[:]
        cluster_boxes_msg.boxes.append(self.cluster_boxes_msg.boxes[index])
        cluster_msg = copy.deepcopy(self.cluster_msg)
        del cluster_msg.cluster_indices[:]
        cluster_msg.cluster_indices.append(self.cluster_msg.cluster_indices[index])

        client_msg = GetFeaturesRequest()
        client_msg.task = 'items'
        client_msg.cloud = self.cloud_msg
        client_msg.image = self.image_msg
        client_msg.cluster = cluster_msg
        client_msg.instance_boxes = instance_boxes_msg
        client_msg.cluster_boxes = cluster_boxes_msg
        client_msg.index = index
        self.get_feature_client(client_msg)

    def service_callback(self, req):
        with self.lock:
            self.call(2)
            self.call(3)

if __name__=='__main__':
    rospy.init_node('service_call_test')
    sample = Sample()
    rospy.spin()
