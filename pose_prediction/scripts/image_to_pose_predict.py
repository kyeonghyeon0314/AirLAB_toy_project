#!/usr/bin/env python3.8

import rospy
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import sys
import os
import argparse

# 모델 경로와 weight 파일 경로 설정
MODEL_PATH = '/home/kimkh/PoseNet-Pytorch/model.py'
WEIGHT_PATH = '/home/kimkh/PoseNet-Pytorch/models_AirLAB'

# PoseNet 모델 불러오기
sys.path.append(os.path.dirname(MODEL_PATH))
from model import model_parser

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

        self.pose_publisher = rospy.Publisher('/predicted_poses', PoseArray, queue_size=10)
        rospy.Subscriber('/gt_images', Image, self.callback)
        rospy.Subscriber('/gt_poses', PoseArray, self.callback_gt_pose)
        self.pose_array_msg = PoseArray()
        self.pose_array_msg.header.frame_id = 'map'
        
        rospy.loginfo(f"Model {model_name} loaded with weights from {weight_path}")

    def callback_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        input_image = self.preprocess_image(cv_image)

        position, rotation = self.predict_pose(input_image)

        rotation = self.normalize_rotation(rotation)

        self.publish_pose(position, rotation)
        
        #디버깅 추가
        if self.gt_poses:
            gt_pose = self.gt_poses.pop(0)
            gt_position = torch.tensor([gt_pose.position.x, gt_pose.position.y, gt_pose.position.z], device=self.device)
            gt_orientation = torch.tensor([gt_pose.orientation.x, gt_pose.orientation.y, gt_pose.orientation.z, gt_pose.orientation.w], device=self.device)
            
            pos_error = torch.norm(position - gt_position).item()
            ori_error = torch.norm(rotation - gt_orientation).item()
            
            print(f"{self.prediction_count}")
            print(f"pos out {position.cpu().numpy()}")
            print(f"ori out {rotation.cpu().numpy()}")
            print(f"pos true {gt_position.cpu().numpy()}")
            print(f"ori true {gt_orientation.cpu().numpy()}")
            print(f"{position.cpu().numpy()}")
            print(f"{gt_position.cpu().numpy()}")
            print(f"{self.prediction_count}th Error: pos error {pos_error:.3f} / ori error {ori_error:.3f}")
            
            self.prediction_count += 1

    
    def callback_gt_pose(self, msg):
        self.gt_poses = msg.poses
    

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processed_image = preprocess(image).unsqueeze(0).to(self.device)
        return processed_image

    def predict_pose(self, image):
        with torch.no_grad():
            position, rotation, _ = self.model(image)
        return position.flatten(), rotation.flatten()

    def normalize_rotation(self, rotation):
        norm = np.linalg.norm(rotation)
        if norm > 1:
            rotation = rotation / norm
        return rotation

    def publish_pose(self, position, rotation):
        pose_msg = Pose()
        pose_msg.position.x = pose[0].item()
        pose_msg.position.y = pose[1].item()
        pose_msg.position.z = pose[2].item()
        pose_msg.orientation.x = pose[3].item()
        pose_msg.orientation.y = pose[4].item()
        pose_msg.orientation.z = pose[5].item()
        pose_msg.orientation.w = pose[6].item()

        self.pose_array_msg.poses.append(pose_msg)
        self.pose_array_msg.header.stamp = rospy.Time.now()
        self.pose_publisher.publish(self.pose_array_msg)
       
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



