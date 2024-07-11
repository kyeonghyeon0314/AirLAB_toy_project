#!/usr/bin/env python3.8

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

def pose_to_marker(pose, marker_id, color):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.pose = pose.pose
    marker.id = marker_id
    return marker

def gt_callback(data):
    global gt_marker_id
    marker = pose_to_marker(data, gt_marker_id, (0, 1, 0))  # Green for GT
    gt_marker_id += 1
    gt_pub.publish(marker)

def pred_callback(data):
    global pred_marker_id
    marker = pose_to_marker(data, pred_marker_id, (1, 0, 0))  # Red for Predicted
    pred_marker_id += 1
    pred_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('rviz_node', anonymous=True)

    gt_pub = rospy.Publisher('/train_marker', Marker, queue_size=10)
    pred_pub = rospy.Publisher('/test_marker', Marker, queue_size=10)

    rospy.Subscriber('/train_pose', PoseStamped, gt_callback)
    rospy.Subscriber('/posenet_pytorch', PoseStamped, pred_callback)

    gt_marker_id = 0
    pred_marker_id = 0

    rospy.spin()
