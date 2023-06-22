cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(example-app)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CUDA_TOOLKIT_ROOT_DIR "3rdParty/cuda-10.1" CACHE PATH "Path to CUDA installed location.")
set(ENV{LD_LIBRARY_PATH} "3rdParty/cuda-10.1/lib64")

set(CMAKE_PREFIX_PATH "3rdParty/libtorch-1.7.1+cu110")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app src/example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(test-model-jit src/test-model-jit.cpp)
target_link_libraries(test-model-jit ${TORCH_LIBRARIES})
set_property(TARGET test-model-jit PROPERTY CXX_STANDARD 17)

add_executable(test-model src/test-model.cpp src/resnet101_impl.cpp src/block.cpp)
target_link_libraries(test-model ${TORCH_LIBRARIES})
target_include_directories(test-model PRIVATE "include")
set_property(TARGET test-model PROPERTY CXX_STANDARD 17)