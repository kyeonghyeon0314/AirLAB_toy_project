#!/usr/bin/env python3.8

import rospy
from geometry_msgs.msg import PoseStamped
import pandas as pd

def publish_gt_poses(gt_file):
    gt_data = pd.read_csv(gt_file, sep=' ', header=None)
    
    gt_pub = rospy.Publisher('/train_pose', PoseStamped, queue_size=10)

    rospy.init_node('data_publish_node', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz

    for idx in range(len(gt_data)):
        gt_pose = PoseStamped()
        
        gt_pose.header.stamp = rospy.Time.now()
        gt_pose.header.frame_id = 'map'
        gt_pose.pose.position.x = gt_data.iloc[idx, 1]
        gt_pose.pose.position.y = gt_data.iloc[idx, 2]
        gt_pose.pose.position.z = gt_data.iloc[idx, 3]
        gt_pose.pose.orientation.x = gt_data.iloc[idx, 4]
        gt_pose.pose.orientation.y = gt_data.iloc[idx, 5]
        gt_pose.pose.orientation.z = gt_data.iloc[idx, 6]
        gt_pose.pose.orientation.w = gt_data.iloc[idx, 7]

        gt_pub.publish(gt_pose)
        rate.sleep()

if __name__ == '__main__':
    gt_file = 'pose_data_test.txt'
    try:
        publish_gt_poses(gt_file)
    except rospy.ROSInterruptException:
        pass
