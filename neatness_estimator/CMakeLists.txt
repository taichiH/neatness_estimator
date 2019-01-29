cmake_minimum_required(VERSION 2.8.3)
project(neatness_estimator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  sensor_msgs
  negomo
  hri_c_flow
  jsk_recognition_msgs
  what_i_see_msgs
  neatness_estimator_msgs
  pcl_conversions
  pcl_ros
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

add_definitions(-std=c++11)

catkin_package(
  CATKIN_DEPENDS
)

add_executable(multi_euclidean_clustering src/multi_euclidean_clustering.cpp)
target_link_libraries(multi_euclidean_clustering ${catkin_LIBRARIES})

# add_executable(battery_state_publisher src/battery_state_publisher.cpp)
# target_link_libraries(battery_state_publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
# add_dependencies(battery_state_publisher battery_state_publisher_generate_messages_cpp)
