# 프로젝트 진행 중 참고 링크
* [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM)
* [PoseNet-Pytorch](https://github.com/youngguncho/PoseNet-Pytorch)

## 진행환경
* ROS1 noetic
* ubuntu 20.04 lts
* opencv 4.2.0
* pcl 1.10
* python 3.8.10
* pytorch
* numpy
* torchvision
* tensorboard
* tensorboardx
* pandas


# 진행 과정
* PosNet-Pytorch 학습
* LeGO-LOAM 설치
* GT 제작(Lidar, SLAM(LeGO-LOAM))
* 수집한 데이터 PoseNet에 학습
* Visual Localization Node 제작
* 실시간으로 GT의 pose정보와 Predict한 pose정보를 Rviz상에서 시각화하기



# PosNet-Pytorch 학습 실습
논문에는 GoogleNet을 활용하는데 ResNet이 기본 선택 되어 있습니다.(논문 발표 당시 GoogleNet이 대중적이라고 합니다.)

tensorboard 사용전 환경변수 설정

1. pip show tensorboard 명령어로 설치 경로를 확인합니다.
2. TensorBoard가 설치된 경로를 시스템 PATH에 추가합니다.(code ~/.bashrc 입력하여 수정)

![tensorboard환경변수설정](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/a4357321-dba8-4ae3-a91e-1a6e551e6ed2)

### -[KingsCollege Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/)
- 기본 셋팅 [ Epoch : 400, lr : 0.0001, dropout rate : 0.5, model 저장 : 50, batch_size : 16, num_epoch_decay : 50(감소율 0.1) ]

[--lr 0.001 --dropout_rate 0.3 --num_epochs 200 --num_epochs_decay 20 --model_save_step 10]  의 설정으로 실행 해보았지만 얻은 수확은 없었습니다.
  
- 기존에 학습 해 두었던 것이 컴퓨터를 바꾸면서 날아가 다시 학습하기에 너무 오래 걸려 향후 학습을 진행하고 우선 초기 학습모델로 진행 하도록 하겠습니다.(24.07.06.sat) , 기존 학다.
* 최적의 파라미터 조합

# Visual Localization Node 제작(미완료)
Test dataset으로 실시간으로 GT의 pose정보와 Predict한 pose정보를 Rviz상에서 시각화하기
	



   

