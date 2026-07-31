cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(go2_deploy)

SET(CMAKE_CXX_STANDARD 17)

set(UNITREE_ROOT $ENV{HOME}/opt/unitree_robotics)

execute_process(
    COMMAND python3 -c "import torch; print(torch.__file__.replace('/__init__.py', '/share/cmake/Torch'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)
if(NOT TORCH_FIND_RESULT EQUAL 0)
    set(TORCH_CMAKE_PATH "$ENV{HOME}/.local/lib/python3.8/site-packages/torch/share/cmake/Torch")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

# Pre-cache nvrtc stub so FindCUDAToolkit (called by Torch) finds it on systems
# without a full CUDA nvrtc install.
find_library(CUDA_nvrtc_LIBRARY NAMES nvrtc
    PATHS $ENV{HOME}/cuda_stub/lib64
    NO_DEFAULT_PATH)
# CUDA_NVRTC_LIB is set by cuda.cmake as set("${CUDA_nvrtc_LIBRARY}" CACHE FILEPATH "")
# without FORCE, so it gets stuck as NOTFOUND if cached from a bad run. Clear it
# each configure so cuda.cmake picks up the freshly-found CUDA_nvrtc_LIBRARY above.
unset(CUDA_NVRTC_LIB CACHE)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
find_library(YAML_CPP_LIB NAMES yaml-cpp REQUIRED)

include_directories(
    include
    ${UNITREE_ROOT}/include
    ${UNITREE_ROOT}/include/ddscxx
    ${TORCH_INCLUDE_DIRS})

add_executable(go2_deploy src/main.cpp)

target_link_libraries(go2_deploy
    ${UNITREE_ROOT}/lib/libunitree_sdk2.a
    ${UNITREE_ROOT}/lib/libddscxx.so
    ${UNITREE_ROOT}/lib/libddsc.so
    ${YAML_CPP_LIB}
    ${TORCH_LIBRARIES}
    Threads::Threads
    )

target_link_options(go2_deploy PRIVATE -Wl,--allow-shlib-undefined)

# PyTorch cmake appends -D_GLIBCXX_USE_CXX11_ABI=0 after definitions,
# so undefine first then force =1 via compile options (appended last).
target_compile_options(go2_deploy PRIVATE -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1)
