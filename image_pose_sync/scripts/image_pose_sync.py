#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse
import numpy as np  # numpy 모듈 추가
import message_filters # 모듈 추가
import tf

class ImagePoseSync:
    def __init__(self, mode):
        # 모드 설정 (train/test)
        self.mode = mode
        
        # 이미지 저장 경로 설정
        if self.mode == 'train':
            self.image_folder = 'images_train'
            self.pose_file_path = 'pose_data_train.txt'
        elif self.mode == 'test':
            self.image_folder = 'images_test'
            self.pose_file_path = 'pose_data_test.txt'
        else:
            rospy.logerr("Mode should be 'train' or 'test'")
            rospy.signal_shutdown("Invalid mode")
            return
        
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        
        # 포즈 데이터 저장 파일 설정
        self.pose_file = open(self.pose_file_path, 'w')
        
        self.bridge = CvBridge()
        
        # ROS 토픽 구독(message_filters 추가)
        odom_sub = message_filters.Subscriber('/integrated_to_init', Odometry)
        image_sub = mesage_filters.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage)
        
        ts = message_filters.ApproximateTimeSynchronizer([odom_sub, image_sub], 10, 0.1)
        ts.registerCallback(self.callback)
        
        self.current_pose = None

    # 이미지 pose 콜백 함수 병합
    def callback(self, odometry, compressedImage):
        self.current_pose = [
            odometry.header.stamp.to_sec(),
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            odometry.pose.pose.position.z,
            odometry.pose.pose.orientation.x,
            odometry.pose.pose.orientation.y,
            odometry.pose.pose.orientation.z,
            odometry.pose.pose.orientation.w
        ]
        
        try:
            np_arr = np.frombuffer(compressedImage.data, np.unit8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            timestamp = compressedImage.header.stamp.to_sec()
            image_filename = f'image_{timestamp:.6f}.jpg'
            image_path = os.path.join(self.image_folder, image_filename)
            cv2.imwrite(image_path, cv_image)
            
            # PoseNet 형식으로 pose 데이터를 저장
            pose = self.current_pose[1:]  # 시간 정보를 제외한 pose 데이터
            self.pose_file.write(f"{image_path} {' '.join(map(str, pose))}\n")
            
            rospy.loginfo("Saved image and pose: %s", image_filename)
        except CvBridgeError as e:
            rospy.logerr("Failed to convert or save image: %s", str(e))

    def save_data(self):
        self.pose_file.close()

if __name__ == '__main__':
    # Argument parser 설정
    parser = argparse.ArgumentParser(description='Image and Pose Sync')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode should be either "train" or "test"')
    args = parser.parse_args()

    rospy.init_node('image_pose_sync')
    sync = ImagePoseSync(args.mode)
    rospy.spin()
    sync.save_data()
  
 
