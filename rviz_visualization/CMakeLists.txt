cmake_minimum_required(VERSION 3.0.2)
project(rviz_visualization)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  rviz
  sensor_msgs
  std_msgs
  tf
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rviz_visualization
  CATKIN_DEPENDS cv_bridge geometry_msgs rospy rviz sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/image_pose_publisher.py
  scripts/image_to_pose_predict.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

