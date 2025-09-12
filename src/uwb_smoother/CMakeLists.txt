cmake_minimum_required(VERSION 3.0.2)
project(uwb_smoother)

## 编译参数
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## 找 ROS 依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

## 找 Eigen3
find_package(Eigen3 REQUIRED)

## catkin 配置
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
  INCLUDE_DIRS include
)

## 头文件路径
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

## 可执行文件
add_executable(distance_matrix_smoother src/distance_matrix_smoother.cpp)
target_link_libraries(distance_matrix_smoother
  ${catkin_LIBRARIES}
  Eigen3::Eigen
)
