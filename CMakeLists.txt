cmake_minimum_required(VERSION 2.8.3)
project(eskf)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  sensor_msgs
  message_filters
  message_generation
  eigen_conversions
  mavros_msgs
)

find_package(cmake_modules)
find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

# include headers
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${mavros_INCLUDE_DIRS}
  include
  include/eskf
)

# build shared library
add_library(ESKF
  src/ESKF.cpp
)

set(RELEVANT_LIBRARIES
  ESKF
  ${catkin_LIBRARIES}
  ${mavros_LIBRARIES}
)

# rigorous error checking
add_definitions("-Wall -Werror -O3 -std=c++11")

# build main filter
add_executable(${PROJECT_NAME}
  src/eskf.cpp
  src/Node.cpp
)
target_link_libraries(${PROJECT_NAME} ${RELEVANT_LIBRARIES})

# ensures messages are generated before hand
add_dependencies(${PROJECT_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
