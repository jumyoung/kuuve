cmake_minimum_required(VERSION 3.0.2)
project(vision_final)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  cv_bridge
  geometry_msgs
  image_transport
  message_generation
  roscpp
  roslib
  rospy
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
add_message_files(
   FILES
   ju_msg.msg
   ju_child.msg
)
generate_messages(
   DEPENDENCIES
   std_msgs 
)


catkin_package(
  INCLUDE_DIRS include
  #LIBRARIES vision_final
  #CATKIN_DEPENDS cv_bridge geometry_msgs image_transport roscpp roslib rospy rospy sensor_msgs std_msgs visualization_msgs
  #DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)




add_executable(vision_acc_node src/vision_acc.cpp)
target_link_libraries(vision_acc_node 
        ${OpenCV_LIBRARIES} 
        ${catkin_LIBRARIES}     
)
