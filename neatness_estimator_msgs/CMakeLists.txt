cmake_minimum_required(VERSION 2.8.3)
project(neatness_estimator_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  geometry_msgs
  std_msgs
  jsk_recognition_msgs
  )

add_message_files(
  FILES
  Histogram.msg
  HistogramArray.msg
  AppearanceDifference.msg
  )

add_service_files(
  FILES
  VisionServer.srv
  GetDifference.srv
  GetMotionPrimitive.srv
  CorrectData.srv
  )

generate_messages(
  DEPENDENCIES
  jsk_recognition_msgs
  geometry_msgs
  std_msgs
  )

catkin_package()
