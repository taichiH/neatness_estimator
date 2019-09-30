cmake_minimum_required(VERSION 2.8.3)
project(neatness_estimator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  sensor_msgs
  jsk_recognition_msgs
  neatness_estimator_msgs
  pcl_conversions
  pcl_ros
  tf
  nodelet
  what_i_see
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

add_definitions(-std=c++11)

catkin_package(
  CATKIN_DEPENDS
  LIBRARIES ${PROJECT_NAME}
  INCLUDE_DIRS include
)

add_library(${PROJECT_NAME} SHARED
  src/multi_euclidean_clustering.cpp)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES})

# add_executable(battery_state_publisher src/battery_state_publisher.cpp)
# target_link_libraries(battery_state_publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
# add_dependencies(battery_state_publisher battery_state_publisher_generate_messages_cpp)
