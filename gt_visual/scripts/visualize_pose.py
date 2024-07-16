#!/usr/bin/env python3.8

import rospy
import tf
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
import time

def parse_gt_file(file_path):
    poses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            image_path = parts[0]
            timestamp = float(image_path.split('_')[-1].split('.')[0] + '.' + image_path.split('_')[-1].split('.')[1])
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            poses.append((timestamp, (px, py, pz, qx, qy, qz, qw)))
    return poses

def publish_pose(publisher, timestamp, pose):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.from_sec(timestamp)
    pose_msg.header.frame_id = 'map'
    pose_msg.pose.position.x = pose[0]
    pose_msg.pose.position.y = pose[1]
    pose_msg.pose.position.z = pose[2]
    pose_msg.pose.orientation.x = pose[3]
    pose_msg.pose.orientation.y = pose[4]
    pose_msg.pose.orientation.z = pose[5]
    pose_msg.pose.orientation.w = pose[6]
    publisher.publish(pose_msg)

def main():
    rospy.init_node('gt_pose_visualizer', anonymous=True)
    pose_publisher = rospy.Publisher('/gt_pose', PoseStamped, queue_size=10)

    gt_file_path = '/home/kimkh/PoseNet-Pytorch/PoseNet/AirLAB/pose_data_test.txt'  # 실제 파일 경로로 수정
    poses = parse_gt_file(gt_file_path)

    rate = rospy.Rate(1)  # 1 Hz

    for timestamp, pose in poses:
        if rospy.is_shutdown():
            break
        publish_pose(pose_publisher, timestamp, pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


