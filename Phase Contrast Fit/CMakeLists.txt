cmake_minimum_required(VERSION 3.0.0)
project(pc_fit VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# eigen library path
include_directories( "C:/Users/VIVY/Downloads/eigen-3.4.0/eigen-3.4.0" )
add_compile_options("-Wa,-mbig-obj")
# remove above two lines on linux systems
add_executable(pc_fit main.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
