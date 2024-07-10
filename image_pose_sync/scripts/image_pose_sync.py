#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse
import numpy as np  # numpy 모듈 추가

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
        
        # ROS 토픽 구독
        rospy.Subscriber('/laser_odom_to_init', Odometry, self.pose_callback)
        rospy.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage, self.image_callback)
        
        # 현재 포즈를 저장하는 변수
        self.current_pose = None
        
    def pose_callback(self, data):
        self.current_pose = [
            data.header.stamp.to_sec(),
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            data.pose.pose.position.z,
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        ]
        
    def image_callback(self, data):
        if self.current_pose is None:
            return
        
        try:
            np_arr = np.frombuffer(data.data, np.uint8)  # np.fromstring -> np.frombuffer로 수정
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            timestamp = data.header.stamp.to_sec()
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
  
 
