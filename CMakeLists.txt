cmake_minimum_required(VERSION 2.8.3)

set(PROJECT_NAME artag_detector)
project(${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)


set(PROJECT_DEPENDENCIES 
  roscpp  
  std_msgs
  sensor_msgs
  mav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

find_package(catkin REQUIRED COMPONENTS
  ${PROJECT_DEPENDENCIES}
)

find_package(Eigen3)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${PROJECT_DEPENDENCIES}
  )

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	include
  include/${PROJECT_NAME}
)

set(SOURCE_FILES
  src/artag_detector.cpp
)

# add_library(${PROJECT_NAME} src/PD_controller.cpp include/PD_controller.hpp )
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} )

add_executable(${PROJECT_NAME}_node src/main_nodes.cpp ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBS})
