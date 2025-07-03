cmake_minimum_required(VERSION 3.0.2)
project(distance)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nlink_parser
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs nlink_parser
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(uwb_matrix_uart src/uwb_matrix_uart.cpp)
add_dependencies(uwb_matrix_uart ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(uwb_matrix_uart
  ${catkin_LIBRARIES}
)

# uwb_matrix_generator
add_executable(uwb_matrix_ros src/uwb_matrix_ros.cpp)
add_dependencies(uwb_matrix_ros ${catkin_EXPORTED_TARGETS})
target_link_libraries(uwb_matrix_ros
  ${catkin_LIBRARIES}
)

add_executable(uwb_matrix_ros_new src/uwb_matrix_ros_tx.cpp)
add_dependencies(uwb_matrix_ros_new ${catkin_EXPORTED_TARGETS})
target_link_libraries(uwb_matrix_ros_new
  ${catkin_LIBRARIES}
)
