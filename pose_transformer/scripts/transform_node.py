#!/usr/bin/env python3.8

import rospy
import argparse
import numpy as np
import tf.transformations as tft
import os

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

def read_poses_from_file(file_path):
    poses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # 빈 줄 무시
                parts = line.strip().split()
                image_file = parts[0]
                pose = [float(part) for part in parts[1:]]  # 이미지 파일명을 제외하고 숫자만 파싱
                poses.append((image_file, pose))
    return poses

def write_poses_to_file(file_path, poses):
    with open(file_path, 'w') as file:
        for image_file, pose in poses:
            pose_str = ' '.join(map(str, pose))
            file.write(f"{image_file} {pose_str}\n")

def transform_poses_in_file(input_file_path, output_file_path):
    poses = read_poses_from_file(input_file_path)
    transformed_poses = [(image_file, transform_pose(pose)) for image_file, pose in poses]
    write_poses_to_file(output_file_path, transformed_poses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform poses in a file.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test.')
    
    args = parser.parse_args()
    
    rospy.init_node('pose_transformer_node')

    base_path = '/home/kimkh/PoseNet-Pytorch/PoseNet/AirLAB/'  
    if args.mode == 'train':
        input_file_path = os.path.join(base_path, 'pose_data_train.txt')
        output_file_path = os.path.join(base_path, 'transformed_pose_data_train.txt')
        rospy.loginfo("Running in train mode.")
    elif args.mode == 'test':
        input_file_path = os.path.join(base_path, 'pose_data_test.txt')
        output_file_path = os.path.join(base_path, 'transformed_pose_data_test.txt')
        rospy.loginfo("Running in test mode.")
    
    transform_poses_in_file(input_file_path, output_file_path)
    rospy.loginfo("Pose transformation completed.")



