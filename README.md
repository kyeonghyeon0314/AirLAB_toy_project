# PoseNet-Pytorch-visual-localization 참고 링크
* https://github.com/RobustFieldAutonomyLab/LeGO-LOAM
* https://github.com/youngguncho/PoseNet-Pytorch

## 진행 환경
* ROS1 noetic
* ubuntu 20.04 lts
* opencv 4.2.0
* pcl 1.10
* python 3.8.10

# 진행 과정
## PosNet-Pytorch 학습
## LeGO-LOAM 컴파일 내 환경과 맞추기
https://github.com/Cascio99/Posenet-Pytorch-Visual-Localization/blob/main/README.md?plain=1
## GT제작
GT 생성을 위해 필요한 것은 이미지와 각 이미지의 pose이고 해당 pose는 LOAM을 통해 수집한다.

Time synchronizer부분의 경우  rosbag info에 보면 lidar를 통해 생성되는 msg의 수와 이미지를 통해 생성되는 msg의 수와 다르기 때문

LOAM을 실행 시켜 보면 각 라이다 msg가 출력되는 부분에서의 pose와 결과로 출력될텐데 이를 통 데이터의 개수가 다른 이미지와 맞추기 위해 사용
