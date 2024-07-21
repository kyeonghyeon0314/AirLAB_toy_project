#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as np
import tf.transformations as tft

def parse_gt_file(gt_file_path):
    data = []
    with open(gt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            image_path = parts[0]
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            data.append((image_path, (px, py, pz, qx, qy, qz, qw)))
    return data

def transform_pose(pose):
    px, py, pz = pose[0], pose[1], pose[2]
    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    
    transformed_px = -pz
    transformed_py = -px
    transformed_pz = py
    
    # 쿼터니언 변환
    quat = [qx, qy, qz, qw]
    rot_quat_x = tft.quaternion_from_euler(np.pi / 2, 0, 0)
    
    # z축을 기준으로 -90도 회전하는 쿼터니언
    rot_quat_z = tft.quaternion_from_euler(0, 0, -np.pi / 2)
    
    # 기존 쿼터니언과 두 회전 쿼터니언을 곱하여 새로운 쿼터니언 생성
    quat_after_x = tft.quaternion_multiply(rot_quat_x, quat)
    transformed_quat = tft.quaternion_multiply(rot_quat_z, quat_after_x)
    
    return transformed_px, transformed_py, transformed_pz, transformed_quat[0], transformed_quat[1], transformed_quat[2], transformed_quat[3]

def main():
    rospy.init_node('image_pose_publisher', anonymous=True)
    image_pub = rospy.Publisher('/gt_images', Image, queue_size=10)
    pose_pub = rospy.Publisher('/gt_poses', PoseArray, queue_size=10)
    rate = rospy.Rate(100)  # Hz
    bridge = CvBridge()
    
    gt_file_path = '/home/kimkh/PoseNet-Pytorch/PoseNet/AirLAB/pose_data_test.txt'  # 실제 파일 경로로 수정
    image_folder_path = '/home/kimkh/PoseNet-Pytorch/PoseNet/AirLAB/images_test'  # 이미지 폴더 경로로 수정

    data = parse_gt_file(gt_file_path)
    
    pose_array_msg = PoseArray()
    pose_array_msg.header.frame_id = 'map'

    for image_path, pose in data:
        if rospy.is_shutdown():
            break
        
        full_image_path = os.path.join(image_folder_path, os.path.basename(image_path))
        
        if not os.path.exists(full_image_path):
            rospy.logwarn(f"Image file {full_image_path} not found.")
            continue
        
        # 이미지 읽기
        cv_image = cv2.imread(full_image_path)
        if cv_image is None:
            rospy.logwarn(f"Failed to read image {full_image_path}.")
            continue
        
        try:
            image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            image_pub.publish(image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {str(e)}")
            continue

        # Pose 메시지 생성 및 발행
        transformed_px, transformed_py, transformed_pz, qx, qy, qz, qw = transform_pose(pose)
        pose_msg = Pose()
        pose_msg.position.x = transformed_px
        pose_msg.position.y = transformed_py
        pose_msg.position.z = transformed_pz
        pose_msg.orientation.x = qx
        pose_msg.orientation.y = qy
        pose_msg.orientation.z = qz
        pose_msg.orientation.w = qw
        
        pose_array_msg.poses.append(pose_msg)
        
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_pub.publish(pose_array_msg)
        
        rospy.loginfo(f"Published image {full_image_path} and pose array.")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass



