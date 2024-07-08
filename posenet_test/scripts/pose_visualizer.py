#!/usr/bin/env python3.8

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import tf2_geometry_msgs
import tf

def callback(image_msg, odom_msg):
    """
    이미지 및 Odometry 메시지가 동기화될 때 호출되는 콜백 함수

    Args:
        image_msg (CompressedImage): 동기화된 압축 이미지 메시지 (현재 사용하지 않음)
        odom_msg (Odometry): 동기화된 Odometry 메시지 (GT pose 정보 포함)
    """
    global tf_buffer, tf_listener

    # GT pose 정보 가져오기 (map 프레임 기준으로 변환)
    try:
        transform = tf_buffer.lookup_transform('map', 'zed_left_camera_frame', odom_msg.header.stamp, rospy.Duration(0.1))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn(f"Failed to get transform: {e}")
        return

    # GT pose 정보를 RViz에 시각화하기 위한 TF 브로드캐스팅
    t = TransformStamped()
    t.header.stamp = odom_msg.header.stamp
    t.header.frame_id = "map"  # 부모 프레임 (map)
    t.child_frame_id = "gt_pose"  # 자식 프레임 (gt_pose)
    t.transform.translation = transform.transform.translation  # 위치 정보 설정
    t.transform.rotation = transform.transform.rotation  # 회전 정보 설정
    tf_broadcaster.sendTransform(t)  # TF 변환 정보 발행

    # GT pose 정보를 RViz에 시각화하기 위한 Marker 생성 및 발행
    marker = Marker()
    marker.header.frame_id = "map"  # 마커의 기준 좌표 프레임 설정
    marker.header.stamp = rospy.Time.now()  # 마커 타임스탬프 설정
    marker.ns = "gt_pose_marker"  # 마커 네임스페이스 설정
    marker.id = 0  # 마커 ID 설정
    marker.type = Marker.ARROW  # 마커 종류 (화살표)
    marker.action = Marker.ADD  # 마커 추가
    marker.pose.position = transform.transform.translation  # 마커 위치 설정
    marker.pose.orientation = transform.transform.rotation  # 마커 자세 설정
    marker.scale.x = 0.2  # 마커 크기 (x축: 화살표 길이)
    marker.scale.y = 0.05  # 마커 크기 (y축: 화살표 두께)
    marker.scale.z = 0.05  # 마커 크기 (z축: 화살표 두께)
    marker.color.a = 1.0  # 마커 투명도
    marker.color.r = 1.0  # 마커 색상 (빨간색)
    marker.color.g = 0.0
    marker.color.b = 0.0
    
    marker_array = MarkerArray()  # MarkerArray 메시지 생성
    marker_array.markers.append(marker)  # 마커 추가
    marker_pub.publish(marker_array)  # 마커 메시지 발행

if __name__ == '__main__':
    rospy.init_node('pose_visualizer')  # ROS 노드 초기화

    tf_buffer = Buffer()  # TF 변환 정보 저장 버퍼 생성
    tf_listener = TransformListener(tf_buffer)  # TF 변환 정보 수신 리스너 생성
    tf_broadcaster = TransformBroadcaster()  # TF 변환 정보 발행 브로드캐스터 생성
    marker_pub = rospy.Publisher('/gt_pose_marker', MarkerArray, queue_size=10)  # 마커 발행 토픽 설정

    # 이미지 토픽 구독 (현재 사용하지 않음)
    image_sub = Subscriber('/zed/left/image_rect_color/compressed', CompressedImage)
    # Odometry 토픽 구독
    pose_sub = Subscriber('/odom', Odometry)

    # ApproximateTimeSynchronizer를 사용하여 이미지와 pose 메시지 동기화
    ats = ApproximateTimeSynchronizer([image_sub, pose_sub], queue_size=10, slop=0.1)  # 0.1초의 시간 오차 허용
    ats.registerCallback(callback)  # 콜백 함수 등록

    rospy.spin()  # ROS 노드 실행 유지


