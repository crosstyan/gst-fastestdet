cmake_minimum_required(VERSION 3.15...3.19)
project(yolo_app)
find_package(PkgConfig REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED OpenCV_DIR)
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
else()
    find_package(OpenCV REQUIRED)
endif()
message("Found OpenCV at" ${OpenCV_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
# pkg_check_modules(Protobuf REQUIRED protobuf)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

if (NOT DEFINED NCNN_INSTALL_DIR)
    set(NCNN_INSTALL_DIR "/usr/local" CACHE STRING "NCNN install directory")
endif ()
set(NCNN_LIBS ${NCNN_INSTALL_DIR}/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS ${NCNN_INSTALL_DIR}/include/ncnn)

set(YOLO_FASTEST_INCLUDE_DIRS include)
set(YOLO_FASTEST_SRC yolo-fastestv2.cpp)
if(NOT DEFINED CLI11_DIR)
    set (CLI11_DIR CLI11 CACHE STRING "CLI11 git repository")
endif()
if(NOT DEFINED FMT_LIB_DIR)
    set (FMT_LIB_DIR fmt CACHE STRING "https://github.com/fmtlib/fmt")
endif()
add_subdirectory(${FMT_LIB_DIR})
add_subdirectory(${CLI11_DIR})
include_directories(
    ${FMT_LIB_DIR}/include
    ${CLI11_DIR}/include 
    ${YOLO_FASTEST_INCLUDE_DIRS} 
    ${NCNN_INCLUDE_DIRS} 
    ${Protobuf_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
add_executable(yolo-fastestv2 ${YOLO_FASTEST_SRC} matrix.pb.cc main.cpp)
target_link_libraries(yolo-fastestv2 PRIVATE 
                        ${NCNN_LIBS} 
                        ${OpenCV_LIBS} 
                        ${Protobuf_LIBRARIES}
                        fmt::fmt)
# target_link_options(yolo-fastestv2 PRIVATE "LINKER: --allow-shlib-undefined" "LINKER: --warn-unresolved-symbols")