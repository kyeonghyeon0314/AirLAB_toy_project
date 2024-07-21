#!/usr/bin/env python3.8

import rospy
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys
import os
import argparse
import tf.transformations as tft

# 모델 경로와 weight 파일 경로 설정
MODEL_PATH = '/home/kimkh/PoseNet-Pytorch/model.py'
WEIGHT_PATH = '/home/kimkh/PoseNet-Pytorch/models_AirLAB'

# PoseNet 모델 불러오기
sys.path.append(os.path.dirname(MODEL_PATH))
from model import model_parser

# Transform pose function
def transform_pose(pose):
    px, py, pz = pose[0], pose[1], pose[2]
    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    
    transformed_px = -pz
    transformed_py = -px
    transformed_pz = py
    
    # 쿼터니언 변환
    quat = [qx, qy, qz, qw]
    rot_quat_x = tft.quaternion_from_euler(np.pi / 2, 0, 0)
    rot_quat_z = tft.quaternion_from_euler(0, 0, -np.pi / 2)
    
    # 기존 쿼터니언과 두 회전 쿼터니언을 곱하여 새로운 쿼터니언 생성
    quat_after_x = tft.quaternion_multiply(rot_quat_x, quat)
    transformed_quat = tft.quaternion_multiply(rot_quat_z, quat_after_x)
    
    rospy.loginfo(f"Original quat: {quat}")
    rospy.loginfo(f"Rotated quat after x: {quat_after_x}")
    rospy.loginfo(f"Transformed quat: {transformed_quat}")
    
    return transformed_px, transformed_py, transformed_pz, transformed_quat[0], transformed_quat[1], transformed_quat[2], transformed_quat[3]

class PosePredictor:
    def __init__(self, model_name, weight_file):
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_parser(model_name).to(self.device)
        
        weight_path = os.path.join(WEIGHT_PATH, weight_file)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        self.pose_publisher = rospy.Publisher('/predict_pose_array', PoseArray, queue_size=10)
        rospy.Subscriber('/image', Image, self.callback)
        self.pose_array_msg = PoseArray()
        self.pose_array_msg.header.frame_id = 'map'
        
        rospy.loginfo(f"Model {model_name} loaded with weights from {weight_path}")

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo("Image received and converted")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        rospy.loginfo(f"Original image shape: {cv_image.shape}")

        input_image = self.preprocess_image(cv_image)
        rospy.loginfo(f"Processed image shape: {input_image.shape}")

        position, rotation = self.predict_pose(input_image)
        rospy.loginfo(f"Predicted position: {position}")
        rospy.loginfo(f"Predicted rotation: {rotation}")

        # Normalizing rotation values if they are out of expected range
        rotation = self.normalize_rotation(rotation)
        rospy.loginfo(f"Normalized rotation: {rotation}")

        transformed_pose = transform_pose(np.concatenate((position, rotation)))
        rospy.loginfo(f"Transformed pose: {transformed_pose}")

        self.publish_pose(transformed_pose)

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processed_image = preprocess(image).unsqueeze(0).to(self.device)
        rospy.loginfo("Image preprocessed")
        return processed_image

    def predict_pose(self, image):
        with torch.no_grad():
            position, rotation, _ = self.model(image)
        position = position.cpu().numpy().flatten()
        rotation = rotation.cpu().numpy().flatten()
        rospy.loginfo(f"Pose predicted: Position {position}, Rotation {rotation}")
        return position, rotation

    def normalize_rotation(self, rotation):
        norm = np.linalg.norm(rotation)
        if norm > 1:
            rotation = rotation / norm
        return rotation

    def publish_pose(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]
        pose_msg.position.z = pose[2]
        pose_msg.orientation.x = pose[3]
        pose_msg.orientation.y = pose[4]
        pose_msg.orientation.z = pose[5]
        pose_msg.orientation.w = pose[6]

        self.pose_array_msg.poses.append(pose_msg)
        self.pose_publisher.publish(self.pose_array_msg)
        rospy.loginfo(f"Published pose: {pose_msg}")

def main():
    parser = argparse.ArgumentParser(description='ROS Node to predict poses from images using a PoseNet model.')
    parser.add_argument('--model', type=str, required=True, help='Model name to use for PoseNet.')
    parser.add_argument('--weight', type=str, required=True, help='Weight file for the model.')
    args = parser.parse_args()

    rospy.init_node('pose_predictor', anonymous=True)

    model_name = args.model
    weight_file = f"{args.weight}_net.pth"  # weight 파일 형식을 맞추기 위해 확장자를 추가

    PosePredictor(model_name, weight_file)
    rospy.spin()

if __name__ == '__main__':
    main()



