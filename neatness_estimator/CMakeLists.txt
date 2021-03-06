cmake_minimum_required(VERSION 2.8.3)
project(neatness_estimator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  geometry_msgs
  sensor_msgs
  jsk_recognition_msgs
  neatness_estimator_msgs
  pcl_conversions
  pcl_ros
  tf
  nodelet
  jsk_pcl_ros
  )

find_package(OpenCV REQUIRED)
find_package(yaml-cpp)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

message("PCL_VERSION: ")
message(${PCL_VERSION})

# dounloads training data
add_custom_target(
  ${PROJECT_NAME}_install_trained_data
  ALL COMMAND ${PROJECT_SOURCE_DIR}/install_trained_data.py
  )


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  yaml-cpp)

add_definitions(-std=c++11)


catkin_package(
  CATKIN_DEPENDS
  LIBRARIES ${PROJECT_NAME}
  INCLUDE_DIRS include
)

add_executable(normal_viewer src/normal_viewer.cpp)
target_link_libraries(normal_viewer
  ${catkin_LIBRARIES}
  )

add_library(${PROJECT_NAME} SHARED
  src/multi_euclidean_clustering.cpp
  src/estimation_module_interface.cpp
  src/objects_feature_extractor.cpp)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
  )

# add_executable(battery_state_publisher src/battery_state_publisher.cpp)
# target_link_libraries(battery_state_publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
# add_dependencies(battery_state_publisher battery_state_publisher_generate_messages_cpp)
