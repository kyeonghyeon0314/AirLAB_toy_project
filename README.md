# PosNet-Pytorch-visual-localization
reference
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

1. tensorboard 설치 경로를 확인합니다.
```
pip show tensorboard
```
2. TensorBoard가 설치된 경로를 시스템 PATH에 추가합니다.
```
code ~/.bashrc
```

![tensorboard환경변수설정](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/a4357321-dba8-4ae3-a91e-1a6e551e6ed2)



### -[KingsCollege Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/)
- 기본 셋팅 [ Epoch : 400, lr : 0.0001, dropout rate : 0.5, model 저장 : 50, batch_size : 16, num_epoch_decay : 50(감소율 0.1) ]
![초기학습 진행 그래프](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/6ef86f09-bf52-42a7-b35b-bd8cc4c7db69)


[--lr 0.001 --dropout_rate 0.3 --num_epochs 200 --num_epochs_decay 20 --model_save_step 10]  의 설정으로 실행 해보았지만 얻은 수확은 없었습니다.
  
- 기존에 학습 해 두었던 것이 컴퓨터를 바꾸면서 날아가 다시 학습하기에 너무 오래 걸려 향후 학습을 진행하고 우선 초기 학습모델로 진행 하도록 하겠습니다.(24.07.06.sat) , 기존 학습 정보
![model174](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/b26a65d9-7701-4e37-998f-7cc3097796f4)
![model274](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/762bd5f9-ece6-4ac8-b696-a34037c735c0)
![model 299](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/7c50400b-0ee1-493f-a104-e7c8815beae3)


# LeGO-LOAM 설치 및 환경과 맞추기
전에 컴파일 과정에서 utillity.h , Cmake.txt 변경 과 같은 여러번의 시도하였지만 해결하지 못하였습니다.

[결국 같이 진행한 분의 도움을 받아 해결하였습니다.](https://github.com/Cascio99/Posenet-Pytorch-Visual-Localization/blob/main/README.md?plain=1)

최종적으로 utillity.h , Cmake.txt , voxel_grid.h 변경

# GT제작(Lidar, SLAM(LeGO-LOAM))

GT를 제작할때 첫시도에 bag 파일에서 나오는 토픽을 구독하여 이미지와 pose 정보를 저장하는 방식으로 노드를 제작하였습니다. 그러나 이 것은 실시간성이 부족하여 로봇을 운행하는데 적합하지 않습니다. 그러나 LeGO-LOAM을 활용하면 실시 포즈 추정 데이터를 활용 할 수 있습니다. 두 방법 모두 기재해 놓도록 하겠습니다.

## bag 파일에서 나오는 토픽 활용(실시간성이 좋지 않음)
1. 패키지 제작
```
catkin_create_pkg synchronizing rospy cv_bridge tf message_filters sensor_msgs
```

* cv_bridge : 이미지 압축을 해제하고 처리하기 위해
* tf : transform 토픽을 subscribe 하고 pose 정보 추출
* message_filters : 시간 동기화를 위한 ApproximateTimeSynchronizer를 제공(일부 오차를 허용)
* sensor_msgs : 이미지 메시지(CompressedImage)를 사용
```
touch image_pose_synchronizer.py  # 파이썬 파일 생성
```
[image_pose_synchronizer.py 생성은 해당 코드 참조](https://github.com/Taemin0707/Regala/blob/main/regala_ros/src/video_stitcher_timeshync.py)



2. 노드 실행 및 gt 제작
* roscore
* 구현한 ROS 노드 실행 : 위에서 제작 노드 실행 (저장하고 싶은 디렉토리에서)
```
rosrun synchronizing image_pose_synchronizer.py
```
* ROS bag 파일 재생 :
```
rosbag play dataset_train.bag --clock(시간 동기화)
rosbag play dataset_test.bag --clock
```
test 데이터셋의 파일 손상이 되어 reindex 진행
```
rosbag reindex dataset_test.bag 
```
* 결과 확인(이미지 와 pose 정보가 동기화되어 저장되었는지 확인)

[LeGO-LOAM 실행 및 GT 제작 영상 첨부 (github_upload - Clipchamp로 제작 (1))](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization.git)

![Screenshot from 2024-07-07 16-58-38](https://github.com/kyeonghyeon0314/PoseNet-Pytorch-visual-localization/assets/132433953/51360090-9b0a-4edd-af69-a9473508292a){: width="60" height="60%"}

PoseNet 학습시 dataloader.py 만들기 용이하게 pose 데이터를 하나의 txt로 만들어 질 수 있도록 image_pose_synchronizer.py를 수정 하였습니다.
```
#기존 코드
    pose_data = f"{position.x},{position.y},{position.z},{orientation.x},{orientation.y},{orientation.z},{orientation.w}\n"
    pose_filename = f"pose_{timestamp.secs}_{timestamp.nsecs}.txt"
    pose_path = os.path.join("poses", pose_filename)  # pose 저장 경로 설정
    with open(pose_path, "w") as f:
        f.write(pose_data)
```
```
#수정 코드
    pose_data = f"images_{test, train 상황에 맞게 수정}/{image_filename} {position.x} {position.y} {position.z} {orientation.x} {orientation.y} {orientation.z} {orientation.w}\n"
    
    
    pose_filename = "poses.txt"
    pose_path = os.path.join("poses", pose_filename)  # pose 저장 경로 설정
    with open(pose_path, "a") as f:  #append 모드로 파일 열기
        f.write(pose_data)
```
![Screenshot from 2024-07-07 23-35-45](https://github.com/kyeonghyeon0314/AirLAB_toy_project/assets/132433953/8eeb2e60-4a4e-4f66-a3ef-bbaad9ae9538)

사진을 보시면 이미지 파일개수와 pose 정보의 개수가 동일 한것을 확인 할 수 있고 PoseNet-Pytorc의 소스코드를 그대로 활용하기 편리해졌습니다. 

+ 학습에는 지장이 없었는데 test 과정에서 문제가 생겨 GT를 좀더 상황에 맞게 설정후 다시 학습을 시켰습니다.(7/09/02:11)


## LeGO-LOAM 활용(실시간성 용이)
패키지 생성 및 빌드
```
cd ~/catkin_ws/src
catkin_create_pkg image_pose_sync rospy std_msgs sensor_msgs nav_msgs
cd ~/catkin_ws
chmod +x /catkin_ws/src/image_pose_sync/scripts/image_pose_sync.py
catkin_make
```

실행 파일 제작
```
mkdir script
touch image_pose_sync.py
```
CMakelist와 package xml 도 수정 하였습니다.
* 실행 과정
```
roscore
roslaunch lego_loam run.launch
rosrun image_pose_sync image_pose_sync.py --mode train
rosbag play dataset_train.bag --clock
```


* 결과
![Screenshot from 2024-07-11 18-12-18](https://github.com/kyeonghyeon0314/AirLAB_toy_project/assets/132433953/b152b840-dfed-466c-a23a-7d4bee9f95ca)
![GT 제작](https://github.com/kyeonghyeon0314/AirLAB_toy_project/assets/132433953/488860d6-3366-49f2-bc20-47401d22895e)
이미지 데이터와 pose 정보의 개수가 일치하여 시간 동기화가 제대로 되어진 것을 확인 하였습니다. 그리고 실시간성이 더 좋아져 데이터 셋의 크기도 증가한것으로 확인됩니다.
* 수정사항
  ```
  # ROS 토픽 구독
  rospy.Subscriber('/integrated_to_init', Odometry, self.pose_callback) # laser_odom_to_init 토픽 변경
  rospy.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage, self.image_callback)
  ```
수정후 train 데이터의 pose와 이미지 데이터의 개수는 11,513개로 줄어 들었습니다.

# 생성한 GT Pose를 Rviz 상에 시각화 하기(Test 데이터 셋을 예시로)

생성된 GT를 PoseNet에 학습 시키기 전에 데이터를 처리를 어떤형식으로 하면 좋을지 알아야 하므로 GT를 먼저 RViz 상에 시각화 하는것이 좋다는 것을 깨달았습니다. 이전에 데이터의 형상을 파악하지 않고 무턱대고 학습만 계속 시켰는데 학습은 제대로 되어지고 있지만 test error가 줄어들지 않는 모습을 보였습니다. 이번 기회에 데이터 전처리 과정이 얼마나 중요한지 몸으로 느낄 수 있는 시간이였습니다. 
우선 LeGO-LOAM을 이용하여 mapping을 하는 과정(아래 사진 참조) 을 보면 고정 축과 동체의 축이 많이 다르 다는 것을 확인 할 수 있습니다. GT의 pose를 동체의 축을 기준으로 다시 맞춰주는 ```transform_pose``` 함수를 추가 하였습니다.
![5](https://github.com/user-attachments/assets/bfd58e3d-2718-4e19-93bb-90a3923c7033)

이제 (```rosrun gt_visual visualize_pose.py```)를 실행해 보았을때 원래의 방향과 다른 방향으로 진행을 하돈 동체가 LeGO-LOAM 상의 동체의 움직임과 동일하게 움직이는 것을 확인 할 수 있습니다.

```
def transform_pose(pose):
    px, py, pz = pose[0], pose[1], pose[2]
    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    
    
    transformed_px = -pz
    transformed_py = -px
    transformed_pz = py
    
    # 쿼터니언 변환
    quat = [qx, qy, qz, qw]
    rot_quat_x = tft.quaternion_from_euler(np.pi / 2, 0, 0)
    
    # z축을 기준으로 90도 회전하는 쿼터니언
    rot_quat_z = tft.quaternion_from_euler(0, 0, -np.pi / 2)
    
    # 기존 쿼터니언과 두 회전 쿼터니언을 곱하여 새로운 쿼터니언 생성
    quat_after_x = tft.quaternion_multiply(rot_quat_x, quat)
    transformed_quat = tft.quaternion_multiply(rot_quat_z, quat_after_x)
    
    return transformed_px, transformed_py, transformed_pz, transformed_quat[0], transformed_quat[1], transformed_quat[2], transformed_quat[3]
```
아래 사진을 보면 축 변환및 회전 후 알맞은 움직임을 보이는 것을 확인 할 수 있습니다.
![1](https://github.com/user-attachments/assets/61b1a4f1-14ff-43b4-92e4-e99e13044780)
![3](https://github.com/user-attachments/assets/7a28c7ed-a0ac-4c52-8167-366a6a8b2279)
![4](https://github.com/user-attachments/assets/b1a7c435-7cbf-4f4a-b7e9-1fb8c0a51853)

어느 한 코너에서 아래 사진과 오류 값이 존재 하지만 전반적으로 GT가 잘 생성 된것을 보아 학습에 지장은 없을 것으로 보입니다.
![2](https://github.com/user-attachments/assets/b871c299-767c-4101-8e1e-d163dd2b1267)

# 취득한 GT로 PoseNet 학습 및 테스트
기존 KingsCollege의 dataset_train.txt 를 보면 3번쨰 라인까지 데이터의 정보에 대한 정보를 담고 있습니다. 해당 부분만 수정하여 학습을 진행하였습니다.
```
class CustomDataset(Dataset):
    def __init__(self, image_path, metadata_path, mode, transform, num_val=100):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[0:]  # 기존은 4번째 부터 
```


## 학습실행
학습 데이터 셋이 1만개 정도 되어 그것의 10%인 1000개의 데이터를 검증 데이터로 지정하였습니다. ```num_val=1000```

추가적으로 모델을 보니 ResNet34를 사용하고 있는데 좀더 높은 정확도를 사용하기 위해 좀더 다층의 ResNet 모델을 추가 하였습니다.(기본 : 34 , 추가 : 50, 101, 152)```dataloader.py``` , ```train.py``` , ```model.py``` 수정

또한, 학습시 ```transform_pose```을 활용 해야 하므로 ```dataloader.py```에 ```transform_pose```함수를 추가하고 알맞게 적용 될수 있도록 설정하였습니다.

* 기존 학습시 짧은 Epoch에 좋은 학습 경과를 보였던 셋팅

train
```
python3 train.py --image_path ./PoseNet/AirLAB --metadata_path ./PoseNet/AirLAB/pose_data_train.txt --model Resnet50 --batch_size 32 --num_epochs 25 --lr 0.001 --num_epochs_decay 5  --model_save_step 5 --num_workers 8
```
![Screenshot from 2024-07-20 18-39-24](https://github.com/user-attachments/assets/d0c0cb43-94ad-4b6e-9f4b-c3373167baf0)
![Screenshot from 2024-07-20 18-33-36](https://github.com/user-attachments/assets/5bf65b73-dc11-433d-aa5b-997b5ad61185)
test
```
python3 test.py --image_path ./PoseNet/AirLAB --metadata_path ./PoseNet/AirLAB/pose_data_test.txt --model Resnet50 --test_model 24
```
축 변환 이후에도 여전히 poss error 가 높게 나온것을 확인 할수 있습니다.
gt가 잘못 생성 된것으로 판단해 test 데이터 셋의 gt만 시각화 하여 확인 하였었지만 train 데이터셋의 gt도 시각화 하여 확인 한 결과 train 데이터셋은 평평하게 형성되지 못한 반번 test 데이터 셋은 평평하게 형성되어 학습이 제대로 이루어 져도 test를 해도 올바를 값을 예측 할 수 없는 것으로 파악됩니다.

test_gt
![Screenshot from 2024-07-21 19-06-51](https://github.com/user-attachments/assets/ac09fb4c-ef58-44a7-82b1-0091b7c51271)
train_gt
![Screenshot from 2024-07-21 19-09-20](https://github.com/user-attachments/assets/302ca7ac-8eb3-4868-9dcf-d333d588523f)
![Screenshot from 2024-07-21 19-38-23](https://github.com/user-attachments/assets/3b850c94-9f29-4a0c-80bb-0d8e2891784e)






# 실시간으로 GT의 pose정보와 예측한 pose값을 Rviz상에서 시각화하기
##Test 데이터 셋의 이미지와 pose 정보 발행
이미지와 pose 정보를 담은 각각의 토픽을 동시간에 같은 주기로 발행 하는 노드 작성하기 실행 파일 ```image_pose_publisher.py```

pose 토픽을 발행 할때는 ```transform_pose```함수를 활용하여 발행 

## 예측한 Pose값을 RViz상에 시각화 노드 제작
발행된 이미지 를 학습된 PoseNet 모델에 입력하여 pose 값 예측 후 예측된 pose 값을 발행
```image_to_pose_pridict.py```
## 패키지 생성
예측
```
cd ~/catkin_ws/src
catkin_create_pkg pose_prediction rospy std_msgs sensor_msgs geometry_msgs
mkdir -p ~/catkin_ws/src/pose_prediction/scripts
cd ~/catkin_ws/src/pose_prediction/scripts
touch image_to_pose_predict.py
```
발행
```
cd ~/catkin_ws/src
catkin_create_pkg image_publisher rospy std_msgs sensor_msgs geometry_msgs
mkdir -p ~/catkin_ws/src/image_publisher/scripts
cd ~/catkin_ws/src/image_publisher/scripts
touch image_pose_publisher.py
```

## launch 형식으로 실행방식 변경
```
cd ~/catkin_ws/src
catkin_create_pkg rviz_visualization rospy std_msgs sensor_msgs geometry_msgs cv_bridge tf rviz
cd ~/catkin_ws/src/rviz_visualization
mkdir launch rviz scripts

cd ~/catkin_ws/src/rviz_visualization/scripts
chmod +x image_pose_publisher.py
chmod +x image_to_pose_predict.py

```

## RViz 상에 시각화
```
roscore
rviz
rosrun pose_prediction image_to_pose_predict.py --model Resnet50 --weight 24
rosrun image_publisher image_pose_publisher.py
```
rviz에 토픽 추가후 관찰

![Screenshot from 2024-07-21 22-19-54](https://github.com/user-attachments/assets/e960c5e7-a357-48b7-8745-07f0172d7b1d)
빨간색이 예측값

현재 예측값이 실제 값과 너무 많은 차이를 보이고 있습니다.



























