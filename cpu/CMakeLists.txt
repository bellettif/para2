cmake_minimum_required(VERSION 3.2)
project(CS267_HW2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(MPI REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${MPI_INCLUDE_PATH})

set(SOURCE_FILES
    #autograder.cpp
    common.cpp
    common.h
    mpi.cpp
    #openmp.cpp
    #pthreads.cpp
    #serial.cpp
    )

add_executable(CS267_HW2 ${SOURCE_FILES} MPIComm.h)

target_link_libraries(CS267_HW2 ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(CS267_HW2 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(CS267_HW2 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

