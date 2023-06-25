cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(example-app)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CUDA_TOOLKIT_ROOT_DIR "3rdParty/cuda-10.1" CACHE PATH "Path to CUDA installed location.")
# set(ENV{LD_LIBRARY_PATH} "3rdParty/cuda-10.1/lib64")

set(Torch_DIR "3rdParty/libtorch-1.7.1+cu101/share/cmake/Torch")

find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED 1.7.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app src/example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(load-model-jit src/load-model-jit.cpp)
target_link_libraries(load-model-jit ${TORCH_LIBRARIES})
set_property(TARGET load-model-jit PROPERTY CXX_STANDARD 17)

add_executable(load-model src/load-model.cpp src/resnet101_impl.cpp src/bottleneck.cpp)
target_link_libraries(load-model ${TORCH_LIBRARIES})
target_include_directories(load-model PRIVATE "include")
set_property(TARGET load-model PROPERTY CXX_STANDARD 17)
