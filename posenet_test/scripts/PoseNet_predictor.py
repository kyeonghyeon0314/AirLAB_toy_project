#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np
import sys

# model.py 파일이 있는 디렉토리를 Python 경로에 추가
sys.path.append('/home/kimkh/PoseNet-Pytorch')

# model.py에서 ResNet 모델 import
from model import ResNet

# 모델 파일 경로
model_path = '/home/kimkh/PoseNet-Pytorch/models_AirLAB/39_net.pth'

# 장치 설정 (CPU 또는 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드 및 설정
base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
model = ResNet(base_model)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((192, 192)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

bridge = CvBridge()

def predict_pose(image, model):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        position, rotation, _ = model(input_batch)

    position = position.squeeze().cpu().numpy()
    rotation = rotation.squeeze().cpu().numpy()
    rospy.loginfo("Predicted position: %s, rotation: %s", position, rotation)
    return position, rotation

def callback(image_msg):
    rospy.loginfo("Image message received")
    try:
        # ROS 압축 이미지를 OpenCV 이미지로 변환
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rospy.loginfo("Converted ROS compressed image to OpenCV image")

        # OpenCV 이미지를 PIL 이미지로 변환
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        rospy.loginfo("Converted OpenCV image to PIL image")

        # 포즈 예측
        position, rotation = predict_pose(pil_image, model)
        rospy.loginfo("Predicted position: %s, rotation: %s", position, rotation)

        # 예측된 position을 이미지에 그리기 (이 예시는 position의 x, y를 사용)
        x, y, z = position
        cv2.circle(cv_image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # 포즈 정보를 발행
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"  # 적절한 프레임 ID를 설정
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]

        rospy.loginfo("Publishing pose: %s", pose_msg)
        pose_pub.publish(pose_msg)

        # 결과 출력
        cv_image_resized = cv2.resize(cv_image, (640, 480))  # 윈도우 크기 조정
        cv2.imshow('PoseNet', cv_image_resized)
        cv2.waitKey(1)

    except CvBridgeError as e:
        rospy.logerr(f"Could not convert ROS Image message to OpenCV image: {e}")
    except Exception as e:
        rospy.logerr(f"Error in callback: {e}")

def listener():
    rospy.init_node('pose_predictor', anonymous=True)
    global pose_pub
    pose_pub = rospy.Publisher('/predicted_pose', PoseStamped, queue_size=10)
    rospy.loginfo("Publisher created for /predicted_pose")
    rospy.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage, callback)
    rospy.loginfo("Subscriber created for /zed/left/image_rect_color/compressed")
    rospy.spin()
    cv2.destroyAllWindows()  # ROS 노드가 종료될 때 OpenCV 윈도우 닫기

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass


