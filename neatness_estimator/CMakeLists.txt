cmake_minimum_required(VERSION 2.8.3)
project(neatness_estimator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  negomo
  hri_c_flow
  jsk_recognition_msgs
  what_i_see_msgs
  neatness_estimator_msgs
  )

catkin_package()
