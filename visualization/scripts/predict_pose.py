#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
import torch
from torchvision import transforms
import cv2
import numpy as np
import sys
import argparse
import os
import threading

# model.py 파일이 있는 디렉토리를 Python 경로에 추가
sys.path.append('/home/kimkh/PoseNet-Pytorch')

# model.py에서 ResNet 및 다른 모델들 import
from model import model_parser

class PosePredictor:
    def __init__(self, model_name, weight_name, topic):
        weight_dir = '/home/kimkh/PoseNet-Pytorch/models_AirLAB'
        model_weights = os.path.join(weight_dir, f'{weight_name}_net.pth')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_parser(model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_weights))
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((192, 192)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.pose_pub = rospy.Publisher('/predicted_pose', PoseStamped, queue_size=10)
        rospy.Subscriber(topic, CompressedImage, self.image_callback)
        self.lock = threading.Lock()
        self.latest_image = None
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.start()

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with self.lock:
            self.latest_image = image

    def process_images(self):
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                with self.lock:
                    image = self.latest_image
                    self.latest_image = None
                pose = self.predict_pose(image)
                self.publish_pose(pose)

    def predict_pose(self, image):
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            position, rotation, _ = self.model(input_batch)
        
        return position.cpu().numpy(), rotation.cpu().numpy()

    def publish_pose(self, pose):
        position, rotation = pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = position[0, 0]
        pose_msg.pose.position.y = position[0, 1]
        pose_msg.pose.position.z = position[0, 2]
        pose_msg.pose.orientation.x = rotation[0, 0]
        pose_msg.pose.orientation.y = rotation[0, 1]
        pose_msg.pose.orientation.z = rotation[0, 2]
        pose_msg.pose.orientation.w = rotation[0, 3]

        self.pose_pub.publish(pose_msg)

if __name__ == "__main__":
    rospy.init_node('predict_pose_node')
    
    parser = argparse.ArgumentParser(description='Predict pose using a pre-trained model')
    parser.add_argument('--model', type=str, required=True, help='Model name to use for prediction')
    parser.add_argument('--weight', type=str, required=True, help='weight name to use for prediction')
    parser.add_argument('--topic', type=str, default='/zed/left/image_rect_color/compressed', help='ROS topic to subscribe to for images')
    args = parser.parse_args(rospy.myargv()[1:])  # rospy.myargv() to get ROS-compatible command line arguments

    predictor = PosePredictor(args.model, args.weight, args.topic)
    rospy.spin()


