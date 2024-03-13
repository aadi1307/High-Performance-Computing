cmake_minimum_required(VERSION 3.8)

project(YourCudaProject)
 
# Enable CUDA language support

enable_language(CUDA)
 
# Find the Eigen library

find_package(Eigen3 REQUIRED NO_MODULE)
 
# Specify your CUDA source files here

add_executable(${PROJECT_NAME} main.cu NonLinearElasticity.cu)
 
# Include directories for Eigen

target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
 
# Link Eigen to your executable

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
 
# Set the CUDA architecture (adjust as needed)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75")
 
# Optional: Set additional NVCC compilation flags

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
