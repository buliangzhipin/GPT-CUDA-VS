cmake_minimum_required(VERSION 3.0.0)
project(GPT-CUDA VERSION 0.1.0)

include(CTest)
enable_testing()

# add_executable(GPT-CUDA main.cpp utility.cpp)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_EXECUTABLE(GPT-CUDA
     GPT-VS-EMPTY-CUDA.cpp utility.cpp init.cpp stdInte.cpp allCUDAFunc.cu
    )
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
