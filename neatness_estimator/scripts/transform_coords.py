#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf
try:
    from what_i_see_msgs.msg import LabeledPose, LabeledPoseArray
except:
    rospy.logerr("please install negomo")
    exit()

class TransformCoords():

    def __init__(self):
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.parent_coords = rospy.get_param('~parent_coords')
        self.pub = rospy.Publisher('~output', LabeledPoseArray, queue_size=1)
        rospy.Subscriber('~input', LabeledPoseArray, self.callback)

    def callback(self, msg):
        pub_box = LabeledPoseArray()
        for index, box in enumerate(msg.poses):
            transformed_box = self.transform_poses(box.pose,
                                                   box.label,
                                                   msg.header.frame_id,
                                                   self.parent_coords)
            transformed_box.confidence = box.confidence
            transformed_box.label = box.label

            pub_box.poses.append(transformed_box)
        pub_box.header = msg.header
        self.pub.publish(pub_box)

    def listen_transform(self, parent_frame, child_frame):
        box = LabeledPose()
        try:
            self.listener.waitForTransform(parent_frame,
                                           child_frame,
                                           rospy.Time(0),
                                           rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform(parent_frame,
                                                         child_frame,
                                                         rospy.Time(0))
            box.pose.position.x = trans[0]
            box.pose.position.y = trans[1]
            box.pose.position.z = trans[2]
            box.pose.orientation.x = rot[0]
            box.pose.orientation.y = rot[1]
            box.pose.orientation.z = rot[2]
            box.pose.orientation.w = rot[3]
            return box
        except:
            rospy.logwarn('cannot lookup transform')
            return box

    def transform_poses(self, pose, label, frame_id, parent):
        self.broadcaster.sendTransform((pose.position.x,
                                        pose.position.y,
                                        pose.position.z),
                                       (pose.orientation.x,
                                        pose.orientation.y,
                                        pose.orientation.z,
                                        pose.orientation.w),
                                       rospy.Time.now(),
                                       label,
                                       frame_id)

        box =  self.listen_transform(parent, label)
        return box

if __name__ == '__main__':
    rospy.init_node("transform_coords")
    tc = TransformCoords()
    rospy.spin()
