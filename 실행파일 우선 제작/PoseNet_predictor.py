#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import torch
from torchvision import transforms
from PIL import Image as PILImage
from cv_bridge import CvBridge
import numpy as np

class PoseNetPredictor:
    def __init__(self):
        self.model = PoseNet()
        self.model.load_state_dict(torch.load('posenet.pth'))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber('/Image', Image, self.image_callback)
        self.pose_pub = rospy.Publisher('/posenet_pytorch', PoseStamped, queue_size=10)
        
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            pil_image = PILImage.fromarray(cv_image)
            image = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(image).squeeze().numpy()
            
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'map'
            pose.pose.position.x = output[0]
            pose.pose.position.y = output[1]
            pose.pose.position.z = output[2]
            pose.pose.orientation.x = output[3]
            pose.pose.orientation.y = output[4]
            pose.pose.orientation.z = output[5]
            pose.pose.orientation.w = output[6]
            
            self.pose_pub.publish(pose)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('posenet_node')
    predictor = PoseNetPredictor()
    rospy.spin()
