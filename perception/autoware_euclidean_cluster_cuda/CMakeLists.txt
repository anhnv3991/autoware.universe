cmake_minimum_required(VERSION 3.14)
project(autoware_euclidean_cluster_cuda)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(PCL REQUIRED)
find_package(CUDA)

if(NOT ${CUDA_FOUND})
  message(WARNING "cuda was not found, so the ${PROJECT_NAME} package will not be built.")
  return()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g -G")
endif()

include_directories(
  include
  SYSTEM
  ${PCL_COMMON_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# cSpell: ignore expt gencode
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr -diag-suppress 20012")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_87,code=sm_87")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=sm_89")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=compute_89")

cuda_add_library(cuda_euclidean_cluster SHARED
  src/cuda_point_cloud2.cu
)

target_include_directories(cuda_euclidean_cluster SYSTEM PRIVATE
  ${autoware_point_types_INCLUDE_DIRS}
  ${cuda_blackboard_INCLUDE_DIRS}
  ${diagnostic_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${rclcpp_components_INCLUDE_DIRS}
  ${rcl_interfaces_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_msgs_INCLUDE_DIRS}
  ${autoware_cuda_utils_INCLUDE_DIRS}
)

ament_auto_add_library(${PROJECT_NAME}_lib SHARED
  lib/euclidean_cluster.cpp
  lib/voxel_grid_based_euclidean_cluster.cpp
  lib/utils.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  ${PCL_LIBRARIES}
  cuda_euclidean_cluster
)

target_include_directories(${PROJECT_NAME}_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_auto_add_library(${PROJECT_NAME}_node_core SHARED
  src/euclidean_cluster_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node_core
  ${PCL_LIBRARIES}
  ${PROJECT_NAME}_lib
)

rclcpp_components_register_node(${PROJECT_NAME}_node_core
  PLUGIN "autoware::euclidean_cluster::EuclideanClusterNode"
  EXECUTABLE euclidean_cluster_node
)

ament_auto_add_library(${PROJECT_NAME}_voxel_grid_node_core SHARED
  src/voxel_grid_based_euclidean_cluster_node.cpp
)
target_link_libraries(${PROJECT_NAME}_voxel_grid_node_core
  ${PCL_LIBRARIES}
  ${PROJECT_NAME}_lib
)

rclcpp_components_register_node(${PROJECT_NAME}_voxel_grid_node_core
  PLUGIN "autoware::euclidean_cluster::VoxelGridBasedEuclideanClusterNode"
  EXECUTABLE voxel_grid_based_euclidean_cluster_node
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_auto_add_gtest(test_voxel_grid_based_euclidean_cluster_fusion
    test/test_voxel_grid_based_euclidean_cluster.cpp
  )
endif()

# Install
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS cuda_euclidean_cluster
  LIBRARY DESTINATION lib
)

ament_auto_package(INSTALL_TO_SHARE
    launch
    config
    test
)
