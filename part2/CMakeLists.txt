cmake_minimum_required(VERSION 2.3)
project(gpu)

SET(INCLUDE_DIR include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(CUDA REQUIRED)

SET(CUDA_PROPAGATE_HOST_FLAGS False)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

SET($INCLUDE_DIRS
        include
)

ADD_DEFINITIONS(${DEFINITIONS})

SET(HEADER_FILES
        ${INCLUDE_DIR}
)

set(
    SOURCE_FILES
    common.cu
    common.h
    physics.cu
    frame_utils.cu
    GPUframe.cu
    gpu.cu
    #serial.cu
)

INCLUDE_DIRECTORIES(${INCLUDE_DIRS})

SET(LIBRARIES
        ${FFTW_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_thrust_LIBRARY}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cufft_LIBRARY}
        ${CUDA_curand_LIBRARY}
)

cuda_add_executable(gpu ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(gpu ${LIBRARIES})