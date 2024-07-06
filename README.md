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
* LeGO-LOAM 컴파일 및 내 환경과 맞추기
* GT 제작(Lidar, SLAM(LeGO-LOAM))
* SLAM의 결과로 나오는 pose정보를 사용, Pose와 이미지의 sync를 어떻게 맞춰야 할지 고민하기 , 수집된 데이터를 활용하여 PoseNet-Pytorch에 학습하기


## PosNet-Pytorch 학습
논문에는 GoogleNet을 활용하는데 ResNet이 기본 선택 되어 있습니다.(논문 발표 당시 GoogleNet이 대중적이라고 합니다.)

### -[KingsCollege Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/)
- 기본 셋팅 [ Epoch : 400, lr : 0.0001, dropout rate : 0.5, model 저장 : 50epoch, batch_size : 16, num_epoch_decay : 50(감소율 0.1) ]
- 기존에 학습 해 두었던 것이 컴퓨터를 바꾸면서 날아가 다시 학습하기에 너무 오래 걸려 향후 학습을 진행하고 우선 초기 학습모델로 진행 하도록 하겠습니다.(24.07.06.sat)

### -AirLab Dataset
각자의 방법으로 학습
,테스트 데이터를 돌려서 실시간 위치를 추정한다.
zed/left/image_rect_color/compressed 사용


## LeGO-LOAM 컴파일 및 환경과 맞추기
전에 컴파일 과정에서 utillity.h , Cmake.txt 변경 과 같은 여러번의 시도하였지만 해결하지 못하였습니다.

[결국 같이 진행한 분의 도움을 받아 해결하였습니다.](https://github.com/Cascio99/Posenet-Pytorch-Visual-Localization/blob/main/README.md?plain=1)

최종적으로 utillity.h , Cmake.txt , voxel_grid.h 변경

## GT제작
lego-loam활용하여 6-Dof의 pose를 각 이미지 별로 매칭하여 취득(Time synchronizer 참고)

GT 생성을 위해 필요한 것은 이미지와 각 이미지의 pose이고 해당 pose는 LOAM을 통해 수집한다.



Time synchronizer부분의 경우  rosbag info에 보면 lidar를 통해 생성되는 msg의 수와 이미지를 통해 생성되는 msg의 수와 다르기 때문

LOAM을 실행 시켜 보면 각 라이다 msg가 출력되는 부분에서의 pose와 결과로 출력될텐데 이를 통 데이터의 개수가 다른 이미지와 맞추기 위해 사용
