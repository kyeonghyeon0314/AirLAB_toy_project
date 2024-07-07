#!/usr/bin/env python3.8

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import tf
import cv2
import numpy as np
import os

def callback(image_msg, odom_msg):
    # 이미지 압축 해제
    bridge = CvBridge()
    image = bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    
    # 이미지 저장
    timestamp = image_msg.header.stamp
    image_filename = f"image_{timestamp.secs}_{timestamp.nsecs}.jpg"
    image_path = os.path.join("images", image_filename)  # 이미지 저장 경로 설정
    cv2.imwrite(image_path, image)

    # pose 정보 추출 및 저장
    pose = odom_msg.pose.pose
    position = pose.position
    orientation = pose.orientation
    
    pose_data = f"{position.x},{position.y},{position.z},{orientation.x},{orientation.y},{orientation.z},{orientation.w}\n"
    pose_filename = f"pose_{timestamp.secs}_{timestamp.nsecs}.txt"
    pose_path = os.path.join("poses", pose_filename)  # pose 저장 경로 설정
    with open(pose_path, "w") as f:
        f.write(pose_data)

rospy.init_node('image_pose_synchronizer')

# 이미지 토픽 구독 (압축된 이미지 토픽)
image_sub = Subscriber('/zed/left/image_rect_color/compressed', CompressedImage)
# pose 토픽 구독 (Odometry 메시지 사용)
odom_sub = Subscriber('/odom', Odometry)  

ats = ApproximateTimeSynchronizer([image_sub, odom_sub], queue_size=10, slop=0.1)
ats.registerCallback(callback)

# 이미지 및 pose 저장 디렉토리 생성
os.makedirs("images", exist_ok=True)
os.makedirs("poses", exist_ok=True)

rospy.spin() #노드 끝날때 까지 실행

