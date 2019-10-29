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
  Neatness.msg
  NeatnessArray.msg
  TargetAndGoal.msg
  TargetAndGoalArray.msg
  DisplayPlan.msg
  DisplayPlanArray.msg
  EdgeHistogram.msg
  EdgeHistogramArray.msg
  Histogram.msg
  HistogramArray.msg
  )

add_service_files(
  FILES
  VisionServer.srv
  GetDisplayFeature.srv
  GetColorHistogram.srv
  DisplayState.srv
  GetDifference.srv
  )

generate_messages(
  DEPENDENCIES
  jsk_recognition_msgs
  geometry_msgs
  std_msgs
  )

catkin_package()
