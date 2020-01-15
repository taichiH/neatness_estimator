#!/usr/bin/env python
# coding: UTF-8

import sys
import numpy as np

import rospy
import tf

from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from neatness_estimator_msgs.srv import VisionServer, VisionServerResponse
from visualization_msgs.msg import Marker, MarkerArray

class NeatnessEstimatorVisionServer():

    def __init__(self):
        mask_rcnn_label_lst = rospy.get_param('~fg_class_names')
        qatm_label_lst = rospy.get_param('~qatm_class_names')
        color_label_lst = ['red']
        self.item_owners = {'conveniShelf1' : [],
                            'conveniShelf2' : [],
                            'container' : []}

        self.label_lst = mask_rcnn_label_lst + qatm_label_lst + color_label_lst

        self.boxes = BoundingBoxArray()

        self.header = None
        self.cluster_boxes_header = None
        self.red_boxes = BoundingBoxArray()
        self.cluster_boxes = BoundingBoxArray()
        self.mask_rcnn_boxes = BoundingBoxArray()
        self.qatm_boxes = BoundingBoxArray()
        self.aligned_instance_boxes = BoundingBoxArray()

        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.shelf_flont_angle = 181.28271996356708

        self.instance_box_callback_cnt = 0
        self.instance_aligned_box_callback_cnt = 0

        self.x_max = 0.8

        self.marker_pub = rospy.Publisher(
            "~output/markers", MarkerArray, queue_size=1)

        rospy.Subscriber(
            "~input_instance_boxes", BoundingBoxArray, self.instance_box_callback)
        rospy.Subscriber(
            "~input_aligned_instance_boxes", BoundingBoxArray, self.aligned_instance_box_callback)
        rospy.Subscriber(
            "~input_cluster_boxes", BoundingBoxArray, self.cluster_box_callback)
        rospy.Subscriber(
            "~input_qatm_pos", BoundingBoxArray, self.qatm_pose_callback)
        rospy.Subscriber(
            "~input_red_boxes", BoundingBoxArray, self.red_box_callback)


        rospy.Service(
            '/display_task_vision_server', VisionServer, self.vision_server)

    def instance_box_callback(self, msg):
        if self.instance_box_callback_cnt == 0:
            print('instance_box_callback')
            self.instance_box_callback_cnt += 1

        self.header = msg.header
        self.mask_rcnn_boxes = msg
        # print('mask_rcnn_boxes size: %s' %(len(self.mask_rcnn_boxes.boxes)))


    def aligned_instance_box_callback(self, msg):
        if self.instance_aligned_box_callback_cnt == 0:
            print('instance_aligned_box_callback')
            self.instance_aligned_box_callback_cnt += 1
        self.aligned_instance_boxes = msg

    def red_box_callback(self, msg):
        for i in range(len(msg.boxes)):
            msg.boxes[i].label = self.label_lst.index('red')
        self.red_boxes = msg

    def cluster_box_callback(self, msg):
        self.cluster_boxes_header = msg.header
        self.cluster_boxes = msg

    def qatm_pose_callback(self, pose_msg):
        self.header = pose_msg.header
        self.qatm_boxes = pose_msg


    def merge_boxes(self, mask_rcnn_boxes, qatm_boxes, red_boxes):
        boxes = BoundingBoxArray()
        boxes.header = self.header
        boxes.boxes = mask_rcnn_boxes.boxes + qatm_boxes.boxes + red_boxes.boxes
        return boxes

    ''' task get_obj_pos '''
    def get_obj_pos(self, req):
        self.boxes = self.merge_boxes(self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False
        has_nearest_item = False

        try:
            nearest_box, has_nearest_item, target_index = self.get_nearest_box(req, self.boxes)
            rospy.loginfo('has_nearest_item = true')
            if has_nearest_item:
                if req.parent_frame == '':
                    rospy.loginfo('req.parent_frame is: empty')
                    transformed_box = nearest_box
                else:
                    rospy.loginfo('req.parent_frame is: %s' %(self.boxes.header.frame_id))
                    transformed_box = self.transform_poses(
                        nearest_box.pose, req.label, self.boxes.header.frame_id, req.parent_frame)

                if not transformed_box:
                    res.status = False
                    return res

                # faile lookup transform
                if transformed_box == BoundingBox():
                    transformed_box = nearest_box
                    transformed_box.header = self.boxes.header
                else:
                    transformed_box.header = self.boxes.header
                    transformed_box.header.frame_id = req.parent_frame
                    transformed_box.dimensions = nearest_box.dimensions

                res.boxes = transformed_box
                res.index = target_index
                res.status = True

        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_multi_obj_pos '''
    def get_multi_obj_pos(self, req):
        print('get_multi_obj_pos')

        self.boxes = self.merge_boxes(
            self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False
        has_items = False

        try:
            multi_boxes, has_items = self.get_multi_boxes(req)
            rospy.loginfo('has_items = true')

            if has_items:
                if req.parent_frame == '':
                    rospy.loginfo('req.parent_frame is: empty')
                else:
                    rospy.loginfo('req.parent_frame is: %s' %(self.boxes.header.frame_id))

                # faile lookup transform
                if multi_boxes != BoundingBoxArray():
                    multi_boxes.header = self.boxes.header
                    multi_boxes.header.frame_id = req.parent_frame
                    res.multi_boxes = multi_boxes.boxes
                    res.status = True
                else:
                    res.status = False
                print(res.boxes)

        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_place_pos '''
    def get_place_pos(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False
        has_nearest_item = False

        can_place = True
        try:
            ref_cluster_box = BoundingBox()

            foremost_item_front = 24 ** 24
            front_edge = False
            for cluster_box in self.cluster_boxes.boxes:
                item_front = cluster_box.pose.position.x -\
                             (cluster_box.dimensions.x * 0.5)
                if item_front == 0.0:
                    continue

                # print(self.label_lst[cluster_box.label], item_front)

                if item_front < foremost_item_front:
                    foremost_item_front = item_front

                if self.label_lst[cluster_box.label] == req.label:
                    ref_cluster_box = cluster_box

                if self.label_lst[cluster_box.label] == 'shelf_flont':
                    rospy.loginfo('detect shelf front edge')
                    front_edge = cluster_box.pose.position.x -\
                                 (cluster_box.dimensions.x * 0.5)

            # when could not detect shelf front, min front item
            if not front_edge:
                rospy.loginfo('use foremost item %f' %(foremost_item_front))
                front_edge = foremost_item_front

            input_boxes = BoundingBoxArray()
            has_item = False
            for box in self.aligned_instance_boxes.boxes:
                if self.label_lst[box.label] == req.label:
                    has_item = True
                    input_boxes.boxes.append(box)

            if not has_item:
                rospy.loginfo('there are no requested item')
                res.can_place = False
                res.status = False
                return res

            sorted_boxes = sorted(
                input_boxes.boxes,
                key = lambda box : box.pose.position.y, reverse=True)

            # calc mean of (y_dim * 0.75)
            dim_mean = 0
            for sorted_box in sorted_boxes:
                dim_mean += (sorted_box.dimensions.y * 0.55)
            dim_mean = dim_mean / len(sorted_boxes)

            rospy.loginfo('dim_mean: %f' %(dim_mean))
            row_boxes_array = []
            row_boxes = BoundingBoxArray()
            if len(sorted_boxes) == 1:
                row_boxes.boxes.append(sorted_boxes[0])
                row_boxes_array.append(row_boxes)
            else:
                for i in range(1, len(sorted_boxes)):
                    row_boxes.boxes.append(sorted_boxes[i-1])
                    y_diff = abs(sorted_boxes[i].pose.position.y -\
                                 sorted_boxes[i-1].pose.position.y)
                    rospy.loginfo('y_diff: %f' %(y_diff))
                    if y_diff > dim_mean:
                        row_boxes_array.append(row_boxes)
                        row_boxes = BoundingBoxArray()

                    print('i, sorted_boxes size', i, len(sorted_boxes) - 1)
                    if i == (len(sorted_boxes) - 1):
                        row_boxes.boxes.append(sorted_boxes[i])
                        row_boxes_array.append(row_boxes)

            rows = len(row_boxes_array)
            rospy.loginfo('%s rows: %d' %(req.label, rows))
            res.rows = rows

            ##### rows visualization
            marker_array = MarkerArray()
            for i, row_boxes in enumerate(row_boxes_array):
                sorted_row_boxes = sorted(
                    row_boxes.boxes,
                    key = lambda box : box.pose.position.x, reverse=False)
                marker = Marker()
                marker.header = self.aligned_instance_boxes.header
                marker.lifetime = rospy.Duration(5)
                marker.ns = 'row_' + str(i)
                marker.action = marker.ADD
                marker.pose.orientation.w = 1.0
                marker.id = 2
                marker.type = marker.LINE_STRIP
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.color.r = 1.0
                marker.color.a = 1.0
                rospy.loginfo('row: %d, size: %d'
                              %(i, len(sorted_row_boxes)))
                for i, box in enumerate(sorted_row_boxes):
                    x = box.pose.position.x - box.dimensions.x
                    y = box.pose.position.y
                    z = box.pose.position.z
                    marker.points.append(Point(x, y ,z))
                    if i == (len(sorted_row_boxes) - 1):
                        x = box.pose.position.x + box.dimensions.x
                        marker.points.append(Point(x, y ,z))

                marker_array.markers.append(marker)

            self.marker_pub.publish(marker_array)
            #####


            target_box = BoundingBox()
            for i, row_boxes in enumerate(row_boxes_array):
                nearest_box, _, _ = self.get_nearest_box(req, row_boxes)
                pos = np.array(
                    [nearest_box.pose.position.x - (nearest_box.dimensions.x * 0.5),
                     nearest_box.pose.position.y,
                     nearest_box.pose.position.z])

                if front_edge:
                    print('front_edge: ', front_edge)
                    edge = np.array(
                        [front_edge,
                         nearest_box.pose.position.y,
                         nearest_box.pose.position.z])
                else:
                    edge = np.array(
                        [ref_cluster_box.pose.position.x -\
                         (ref_cluster_box.dimensions.x * 0.5),
                         nearest_box.pose.position.y,
                         nearest_box.pose.position.z])

                self.broadcaster.sendTransform(
                    (pos[0], pos[1], pos[2]),
                    (0,0,0,1),
                    rospy.Time.now(), 'pos_' + str(i),
                    self.cluster_boxes.header.frame_id)
                self.broadcaster.sendTransform(
                    (edge[0], edge[1], edge[2]),
                    (0,0,0,1),
                    rospy.Time.now(), 'edge_' + str(i),
                    self.cluster_boxes.header.frame_id)

                dist_vec = edge - pos
                target_pos = nearest_box
                rospy.loginfo('dist: %f' %(np.linalg.norm(dist_vec)))
                if np.linalg.norm(dist_vec) > req.can_place_thresh:
                    rospy.loginfo('can place')
                    target_pos = pos + (dist_vec * 0.5)
                    self.broadcaster.sendTransform(
                        (target_pos[0], target_pos[1], target_pos[2]),
                        (0,0,0,1),
                        rospy.Time.now(), 'target_pos_' + str(i),
                        self.cluster_boxes.header.frame_id)

                    target_box = nearest_box
                    target_box.pose.position.x = target_pos[0]
                    target_box.pose.position.y = target_pos[1]
                    target_box.pose.position.z = target_pos[2]
                    can_place = True
                    break
                else:
                    rospy.loginfo('cannot place')
                    target_pos = np.array(
                        [nearest_box.pose.position.x,
                         nearest_box.pose.position.y,
                         nearest_box.pose.position.z])
                    target_box = nearest_box
                    target_box.pose.position.x = target_pos[0]
                    target_box.pose.position.y = target_pos[1]
                    target_box.pose.position.z = target_pos[2]
                    can_place = False

            res.boxes = target_box
            res.index = target_box.label
            res.can_place = can_place
            res.status = True

        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res


    ''' task get_cluster_items '''
    def get_cluster_items(self, req):
        print('get_cluster_items')

        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        try:
            shelf_height = req.target.z
            for cluster_box in self.cluster_boxes.boxes:
                if cluster_box.label == int(self.label_lst.index('shelf_flont')):
                    if cluster_box.pose.position.z + (cluster_box.dimensions.z * 0.5) > shelf_height:
                        shelf_height = cluster_box.pose.position.z

            rospy.loginfo('shelf_height: %f' %(shelf_height))
            sorted_boxes = sorted(
                self.cluster_boxes.boxes, key = lambda box : box.pose.position.y, reverse=True)

            extracted_boxes = BoundingBoxArray()
            for box in sorted_boxes:
                if box.pose.position.z > shelf_height and\
                   -0.8 < box.pose.position.y and box.pose.position.y < 0.8 and\
                   box.pose.position.x < self.x_max:
                    self.broadcaster.sendTransform(
                        (box.pose.position.x, box.pose.position.y, box.pose.position.z),
                        (box.pose.orientation.x, box.pose.orientation.y,
                         box.pose.orientation.z, box.pose.orientation.w),
                        rospy.Time.now(), self.label_lst[box.label] + '_cluster',
                        self.cluster_boxes.header.frame_id)
                    extracted_boxes.boxes.append(box)

            res.multi_boxes = extracted_boxes.boxes
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        res.status = True
        return res

    ''' task get_mis_place_item '''
    def get_mis_place_item(self, req):
        print('get_mis_place_item')

        self.boxes = self.merge_boxes(
            self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        try:
            item = req.label # cluster item label (string)

            print(len(self.cluster_boxes.boxes))

            x_range = {'min': 0, 'max':0}
            y_range = {'min': 0, 'max':0}
            z_range = {'min': 0, 'max':0}
            for cluster_box in self.cluster_boxes.boxes:
                if self.label_lst[cluster_box.label] == req.label:
                    x_range['min'] = cluster_box.pose.position.x - (cluster_box.dimensions.x * 0.5)
                    x_range['max'] = cluster_box.pose.position.x + (cluster_box.dimensions.x * 0.5)
                    y_range['min'] = cluster_box.pose.position.y - (cluster_box.dimensions.y * 0.5)
                    y_range['max'] = cluster_box.pose.position.y + (cluster_box.dimensions.y * 0.5)
                    z_range['min'] = cluster_box.pose.position.z - (cluster_box.dimensions.z * 0.5)
                    z_range['max'] = cluster_box.pose.position.z + (cluster_box.dimensions.z * 0.5)
                    break

            # self.broadcaster.sendTransform(
            #     (x_range['min'], y_range['min'], z_range['min']),
            #     (0,0,0,1),
            #     rospy.Time.now(), 'min_pos',
            #     self.cluster_boxes_header.frame_id)

            # self.broadcaster.sendTransform(
            #     (x_range['max'], y_range['max'], z_range['max']),
            #     (0,0,0,1),
            #     rospy.Time.now(), 'max_pos',
            #     self.cluster_boxes_header.frame_id)

            res.has_item = False
            for box in self.boxes.boxes:
                if self.label_lst[box.label] == req.label:
                    continue

                transformed_box = self.transform_poses(
                    box.pose, self.label_lst[box.label],
                    self.boxes.header.frame_id, self.cluster_boxes_header.frame_id)
                center = [transformed_box.pose.position.x,
                          transformed_box.pose.position.y,
                          transformed_box.pose.position.z]

                if (x_range['min'] < center[0] and center[0] < x_range['max']) and\
                   (y_range['min'] < center[1] and center[1] < y_range['max']) and\
                   (z_range['min'] < center[2] and center[2] < z_range['max']):
                    message = 'get mis-place item,  {} inside {} cluster'.format(
                        self.label_lst[box.label], req.label)
                    rospy.loginfo(message)
                    res.has_item = True
                    transformed_box.label = box.label
                    res.boxes = transformed_box
                    res.message = message
                    break
            if not res.has_item:
                message = 'there are no mis place item'
                rospy.loginfo(message)
                res.message = message

            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task check_item_stock '''
    def check_item_stock(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        x_max = self.x_max # shelf depth
        z_min = 0.85 # shelf height
        rospy.loginfo('x_max: %f,  z_min: %f' %(x_max, z_min))

        try:
            spot_name = req.label

            # when update stock is true, update stock
            if req.flag:
                self.item_owners[spot_name] = []
                for box in self.aligned_instance_boxes.boxes:
                    if box.pose.position.x > x_max or \
                       box.pose.position.z < z_min:
                        continue

                    label_name = self.label_lst[box.label]
                    if not label_name in self.item_owners[spot_name]:
                        self.item_owners[spot_name].append(label_name)

            res.labels = self.item_owners[spot_name]
            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res


    ''' task get_item_belonging '''
    def get_item_belonging(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        print(self.item_owners)
        print('req label: ', req.label)

        nearest_owner_spot = ''
        min_distance = 24 ** 24
        try:
            for owner_spot, owner_contents in zip(self.item_owners.keys(), self.item_owners.values()):
                print('owner_spot: ', owner_spot)
                print('owner_contents: ', owner_contents)

                if owner_spot == 'container' or owner_spot == req.spot:
                    continue

                if req.label in owner_contents:
                    box = listen_transform(req.spot, owner_spot)
                    if not box:
                        rospy.logwarn('failed to lookup transform in get_item_belonging')
                        res.status = False
                        return res

                    distance = np.linalg.norm(
                        np.array([box.pose.position.x,
                                  box.pose.position.y,
                                  box.pose.position.z]))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_owner_spot = owner_spot

            rospy.loginfo(nearest_owner_spot)
            res.message = nearest_owner_spot
            rospy.loginfo('item belong in %s' %(res.message))
            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_container_stock '''
    def get_container_stock(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False
        res.has_item = False
        try:
            if req.label in self.item_owners['container']:
                rospy.loginfo('%s has %s' %('container', req.label))
                res.has_item = True

            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res


    ''' task check_rough_pose_fitting '''
    def check_rough_pose_fitting(self, req):
        print('check_rough_pose_fitting')

        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        try:
            self.boxes = self.merge_boxes(
                self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

            target_quaternion = []
            ref_quaternion = []

            target_box = self.boxes.boxes[req.label_index]
            if self.label_lst[target_box.label] != req.label:
                rospy.logwarn(
                    '[error] target box label: %s' %(self.label_lst[target_box.label]))
                res.status = False
                return res
            target_quaternion = np.array([target_box.pose.orientation.x,
                                          target_box.pose.orientation.y,
                                          target_box.pose.orientation.z,
                                          target_box.pose.orientation.w])


            ref_box = self.boxes.boxes[req.ref_label_index]
            if self.label_lst[ref_box.label] != req.label:
                rospy.logwarn(
                    '[error] ref box label: %s' %(self.label_lst[ref_box.label]))
                res.status = False
                return res
            ref_quaternion = np.array([ref_box.pose.orientation.x,
                                       ref_box.pose.orientation.y,
                                       ref_box.pose.orientation.z,
                                       ref_box.pose.orientation.w])

            res.pose_dist = np.dot(target_quaternion, ref_quaternion)

            print('target: ', target_quaternion)
            print('ref: ', ref_quaternion)
            print('dot: ', res.pose_dist)
            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res


    ''' task get_distance_from_shelf_front '''
    def get_distance_from_shelf_front(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()

        try:
            self.boxes = self.merge_boxes(
                self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

            has_shelf = False
            for box in self.boxes.boxes:
                if box.label == int(self.label_lst.index('shelf_flont')):
                    has_shelf = True
                    shelf_front = np.array([box.pose.position.x, box.pose.position.y])
                    shelf_edge = np.array([box.pose.position.x, box.pose.position.y]) +\
                                 np.array([box.dimensions.x * 0.5, box.dimensions.y * 0.5])

            if not has_shelf:
                rospy.logwarn("cannot find shelf front")
                res.status = False
                return res

            nearest_box, has_nearest_item, _ = self.get_nearest_box(req, self.boxes)
            if has_nearest_item:
                target_vec = np.array([nearest_box.pose.position.x, nearest_box.pose.position.y])
                target_vec = target_vec - shelf_front
                shelf_vec = shelf_edge - shelf_front
                theta = np.arccos(
                    np.dot(target_vec, shelf_vec) / np.linalg.norm(target_vec) * np.linalg.norm(shelf_vec))
                theta = np.deg2rad(180) - theta if theta > np.deg2rad(90) else theta
                distance  = np.linalg.norm(target_vec) * np.sin(theta)

                if shelf_front is not None and target_vec is not None:
                    res.pulling_dist = distance
                    res.status = True
                else:
                    res.status = False
            else:
                res.status = False
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_distance_from_shelf_front_simple '''
    def get_distance_from_shelf_front_simple(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()

        try:
            self.boxes = self.merge_boxes(
                self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

            has_shelf = False
            for box in self.boxes.boxes:
                if box.label == int(self.label_lst.index('shelf_flont')):
                    has_shelf = True
                    shelf_front = box.pose.position.x

            if not has_shelf:
                rospy.logwarn("cannot find shelf front")
                res.status = False
                return res

            nearest_box, has_nearest_item, _ = self.get_nearest_box(req, self.boxes)
            if has_nearest_item:
                target = nearest_box.pose.position.x - box.dimensions.x * 0.5
                distance  = target - shelf_front

                if shelf_front is not None and target is not None:
                    res.pulling_dist = distance
                    res.status = True
                else:
                    res.status = False
            else:
                res.status = False
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res


    ''' task get_distance_between_two_items '''
    def get_items_distance(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()

        has_item = False
        has_ref_item = False

        try:
            self.boxes = self.merge_boxes(
                self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)

            left_side = -sys.maxsize # left side of right item
            right_side = sys.maxsize # right side of left item

            for box in self.boxes.boxes:
                if box.pose.position.x == 0 and box.pose.position.y == 0 and box.pose.position.z == 0:
                    continue

                if box.label == int(self.label_lst.index(req.ref_label)):
                    has_ref_item = True
                    tmp_left_side = box.pose.position.y + box.dimensions.y * 0.5
                    if tmp_left_side > left_side:
                        left_side = tmp_left_side

            for box in self.boxes.boxes:
                if box.label == int(self.label_lst.index(req.label)):
                    has_item = True
                    tmp_right_side = box.pose.position.y - box.dimensions.y * 0.5
                    if tmp_right_side < right_side:
                        right_side = tmp_right_side

            if right_side == sys.maxsize or left_side == -sys.maxsize \
               or not has_item or not has_ref_item:
                res.status = False
                res.filling_dist = 0
                return res

            res.filling_dist = right_side - left_side
            res.status = True
        except:
            res.status = False
            import traceback
            traceback.print_exc()

        return res

    ''' task get_empty_space '''
    def get_empty_space(self, req):
        rospy.loginfo(req.task)
        rospy.logerr('get_empty_space is not used')
        res.status = False
        return res

    ''' task get_shelf_map_rotation '''
    def get_shelf_map_rotation(self, req):
        rospy.loginfo(req.task)
        res = VisionServerResponse()
        res.status = False

        # overwrite request label
        req.label = 'shelf_flont'

        get_obj_pos_res = self.get_obj_pos(req)
        if get_obj_pos_res.status == True:

            qua = (get_obj_pos_res.boxes.pose.orientation.x,
                   get_obj_pos_res.boxes.pose.orientation.y,
                   get_obj_pos_res.boxes.pose.orientation.z,
                   get_obj_pos_res.boxes.pose.orientation.w)
            e = tf.transformations.euler_from_quaternion(qua)
            # print('rpy')
            # print(np.rad2deg(e[0]), np.rad2deg(e[1]), np.rad2deg(e[2]))

            res.boxes.pose.position.x = e[0]
            res.boxes.pose.position.y = e[1]
            res.boxes.pose.position.z = e[2]

            print(np.rad2deg(e[2]))

            if np.rad2deg(e[2]) == 0:
                res.status == False
                return res

            if np.rad2deg(e[2]) < 0:
                res.boxes.pose.orientation.z = -self.shelf_flont_angle - np.rad2deg(e[2])
            else:
                res.boxes.pose.orientation.z = self.shelf_flont_angle - np.rad2deg(e[2])

            print('angle diff: ', res.boxes.pose.orientation.z)
            res.status = True

        return res

    def get_nearest_box(self, req, boxes):
        distance = 100
        has_request_item = False
        target_index = 0
        nearest_box = BoundingBox()
        for index, box in enumerate(boxes.boxes):
            if box.pose.position.x == 0 or \
               box.pose.position.y == 0 or \
               box.pose.position.z == 0:
                rospy.logwarn('boxes has (0, 0, 0) position box')
                continue

            if self.label_lst[box.label] == req.label:
                has_request_item = True
                ref_point = np.array([box.pose.position.x + (box.dimensions.x * 0.5),
                                      box.pose.position.y + (box.dimensions.y * 0.5),
                                      box.pose.position.z + (box.dimensions.z * 0.5)])

                target_point = np.array([req.target.x,
                                         req.target.y,
                                         req.target.z])

                if np.linalg.norm(ref_point - target_point) < distance:
                    nearest_box.pose = box.pose
                    nearest_box.dimensions = box.dimensions
                    nearest_box.label = box.label
                    target_index = index
                    distance = np.linalg.norm(ref_point - target_point)

        return nearest_box, has_request_item, target_index

    def get_multi_boxes(self, req):
        distance = 100
        has_request_item = False
        multi_boxes = BoundingBoxArray()
        self.boxes = self.merge_boxes(
            self.mask_rcnn_boxes, self.qatm_boxes, self.red_boxes)
        for index, box in enumerate(self.boxes.boxes):
            if box.pose.position.x == 0 or \
               box.pose.position.y == 0 or \
               box.pose.position.z == 0:
                rospy.logwarn('boxes has (0, 0, 0) position box')
                continue

            if self.label_lst[box.label] == req.label:
                has_request_item = True
                transformed_box = self.transform_poses(
                    box.pose, req.label, self.boxes.header.frame_id, req.parent_frame)
                multi_boxes.boxes.append(transformed_box)

        return multi_boxes, has_request_item

    def listen_transform(self, parent_frame, child_frame):
        rospy.loginfo('lookup')
        box = BoundingBox()
        try:
            self.listener.waitForTransform(
                parent_frame, child_frame, rospy.Time(0), rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform(
                parent_frame, child_frame, rospy.Time(0))

            box.pose.position = Point(trans[0], trans[1], trans[2])
            box.pose.orientation = Quaternion(rot[0], rot[1], rot[2], rot[3])
            return box
        except:
            rospy.logwarn('cannot lookup transform')
            return False

    def transform_poses(self, pose, label, frame_id, parent):
        rospy.loginfo('transform_poses')
        rospy.loginfo('broadcast')
        self.broadcaster.sendTransform(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            rospy.Time.now(), label, frame_id)
        box =  self.listen_transform(parent, label)
        return box

    def vision_server(self, req):
        if req.task == 'get_obj_pos':
            return self.get_obj_pos(req)

        elif req.task == 'get_multi_obj_pos':
            rospy.loginfo(req.task)
            return self.get_multi_obj_pos(req)

        elif req.task == 'get_place_pos':
            return self.get_place_pos(req)

        elif req.task == 'get_distance':
            rospy.loginfo(req.task)
            return self.get_distance_from_shelf_front_simple(req)

        elif req.task == 'get_items_distance':
            rospy.loginfo(req.task)
            return self.get_items_distance(req)

        elif req.task == 'get_empty_space':
            rospy.loginfo(req.task)
            return self.get_empty_space(req)

        elif req.task == 'get_shelf_map_rotation':
            rospy.loginfo(req.task)
            return self.get_shelf_map_rotation(req)

        elif req.task == 'get_cluster_items':
            rospy.loginfo(req.task)
            return self.get_cluster_items(req)

        elif req.task == 'get_mis_place_item':
            rospy.loginfo(req.task)
            return self.get_mis_place_item(req)

        elif req.task == 'get_item_belonging':
            rospy.loginfo(req.task)
            return self.get_item_belonging(req)

        elif req.task == 'get_container_stock':
            rospy.loginfo(req.task)
            return self.get_container_stock(req)

        elif req.task == 'check_rough_pose_fitting':
            rospy.loginfo(req.task)
            return self.check_rough_pose_fitting(req)

        elif req.task == 'check_item_stock':
            rospy.loginfo(req.task)
            return self.check_item_stock(req)



if __name__ == "__main__":
    rospy.init_node("display_task_vision_server")
    vision_server = NeatnessEstimatorVisionServer()
    rospy.spin()
