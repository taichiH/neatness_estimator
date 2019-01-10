cmake_minimum_required(VERSION 2.8.3)
project(neatness_estimator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  message_generation
  negomo
  hri_c_flow
  jsk_recognition_msgs
  what_i_see_msgs
  )

add_message_files(
  FILES
  Neatness.msg
  )

add_service_files(
  FILES
  VisionServer.srv
  )

generate_messages(
  DEPENDENCIES
  jsk_recognition_msgs
  geometry_msgs
  )

catkin_package()
