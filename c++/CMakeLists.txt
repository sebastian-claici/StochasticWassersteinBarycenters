cmake_minimum_required(VERSION 3.0)
project(Barycenters)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -ggdb -fopenmp")

# Eigen
set( EIGEN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/Eigen" )
if(NOT (EXISTS ${EIGEN_INCLUDE_DIR}))
  message(FATAL_ERROR "Eigen was not found!")
endif()
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

# LBFGS
set( LBFGS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/LBFGS" )
if(NOT (EXISTS ${LBFGS_INCLUDE_DIR}))
  message(FATAL_ERROR "LBFGS was not found!")
endif()
include_directories(SYSTEM ${LBFGS_INCLUDE_DIR})
include_directories(SYSTEM "${LBFGS_INCLUDE_DIR}/LBFGS")

# OpenCV
find_package( OpenCV REQUIRED )
if (OpenCV_FOUND)
  include_directories(SYSTEM ${OPENCV_INCLUDE_DIRS})
endif()

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS program_options REQUIRED)

# Local libraries
include_directories( "${PROJECT_SOURCE_DIR}/src" )
add_subdirectory(src)

# Sources list
set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  )

add_executable(${PROJECT_NAME} ${SOURCES})
if (OpenCV_FOUND)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()
