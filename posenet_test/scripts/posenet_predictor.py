#!/usr/bin/env python3.8

import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image as PILImage  # PIL을 이용한 이미지 변환
from lib.model import PoseNet
from lib.transformations import quaternion_from_matrix
from geometry_msgs.msg import PoseStamped  # ROS PoseStamped 메시지
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ArgumentParser 설정
parser = argparse.ArgumentParser(description='PoseNet ROS node')
parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights')
parser.add_argument('--input_type', type=str, default='rgb', help='Input type for PoseNet (default: rgb)')
opt = parser.parse_args()

# ROS 노드 초기화 및 Publisher 생성
rospy.init_node('posenet_predictor')
pose_pub = rospy.Publisher('/predicted_pose', PoseStamped, queue_size=10)

# PoseNet 모델 로드 및 GPU 설정
weights_path = opt.model
model = PoseNet(opt.input_type).cuda()
model.load_state_dict(torch.load(weights_path))
model.eval()

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

bridge = CvBridge()

def image_callback(image_msg):
    global model, pose_pub

    # 이미지 전처리
    cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    pil_image = PILImage.fromarray(cv_image)
    image = transform(pil_image).unsqueeze(0)

    # GPU로 이동
    image = Variable(image).cuda()

    # PoseNet 모델 추론
    pose_q, pose_x = model(image)
    pose_q = pose_q.cpu().data.numpy()[0]  # (4,) 형태로 변환
    pose_x = pose_x.cpu().data.numpy()[0]

    # 예측 pose 정보 발행 (PoseStamped 메시지)
    pose_stamped = PoseStamped()
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.header.frame_id = "map"  # 맵 프레임 이름으로 설정
    pose_stamped.pose.position.x = pose_x[0]
    pose_stamped.pose.position.y = pose_x[1]
    pose_stamped.pose.position.z = pose_x[2]
    pose_stamped.pose.orientation.x = pose_q[0]
    pose_stamped.pose.orientation.y = pose_q[1]
    pose_stamped.pose.orientation.z = pose_q[2]
    pose_stamped.pose.orientation.w = pose_q[3]
    pose_pub.publish(pose_stamped)

# 이미지 토픽 구독
image_sub = rospy.Subscriber('/image', Image, image_callback)  # '/image' 토픽 구독

rospy.spin()


