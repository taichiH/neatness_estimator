#!/usr/bin/env python

import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import rospy
import rospkg
from neatness_estimator_msgs.srv import GetMotionPrimitive, GetMotionPrimitiveResponse

class GetMotionPrimitiveServer():

    def __init__(self):
        self.motion_lst = ['rot', 'trans', 'ok', 'other']
        self.label_lst = rospy.get_param('~fg_class_names')
        self.model_path = rospy.get_param(
            '~model_path',
            os.path.join(
                rospkg.RosPack().get_path('neatness_estimator'),
                'trained_data/sample.csv'))

        self.target_item = rospy.get_param('target_item', '')
        self.classifier = None
        rospy.loginfo('model_path: %s' %(self.model_path))
        self.generate_model(self.model_path, self.target_item)

        rospy.Service('~classify', GetMotionPrimitive, self.service_callback)

    def generate_model(self, model_path, target_item):
        self.classifier = RandomForestClassifier(max_depth=2, random_state=0)
        test_data = [] #x
        trained_data = [] #y
        labels = []
        with open(self.model_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rospy.loginfo('target_item: %s' %(target_item))
            for i, row in enumerate(reader):
                if i == 0:
                    labels = row
                    continue

                if target_item == '':
                    test_data.append(map(lambda x : float(x), row[1:4]))
                    trained_data.append(float(row[0]))
                else:
                    idx =self.label_lst.index(target_item)
                    if int(row[4]) != idx:
                        continue
                    test_data.append(map(lambda x : float(x), row[1:4]))
                    trained_data.append(float(row[0]))

        rospy.loginfo('test_data length: %s' %(len(test_data)))
        rospy.loginfo('trained_data length: %s' %(len(trained_data)))

        self.classifier.fit(np.array(test_data), np.array(trained_data))

    def run(self, target_data):
        if self.classifier is None:
            return None

        print('target_data', target_data)
        # random forest
        motion_class = self.classifier.predict(np.array([target_data]))
        motion_class = int(motion_class[0])
        return self.motion_lst[motion_class]

    def service_callback(self, req):
        res = GetMotionPrimitiveResponse()

        if req.update_model:
            rospy.loginfo('update model')
            self.generate_model(self.model_path, req.target_item)

        motion_primitives = []
        for color, geometry, group in zip(
                req.color_distance,
                req.geometry_distance,
                req.group_distance):
            target_data = np.array([color, geometry, group], dtype=np.float64)

            motion_primitive = self.run(target_data)
            if motion_primitive is None:
                rospy.logwarn('failed to motion classification')
                continue

            motion_primitives.append(motion_primitive)

        res.motions = motion_primitives
        res.success = True
        return res

if __name__=='__main__':
    rospy.init_node('get_motion_primitive_server')
    get_motion_primitive_server = GetMotionPrimitiveServer()
    rospy.spin()
