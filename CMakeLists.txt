cmake_minimum_required(VERSION 3.10.0)
project(Learning-Vulkan)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(Vulkan)
if (Vulkan_FOUND)
    message(STATUS "Vulkan Packge Include : " ${Vulkan_INCLUDE_DIR})
    message(STATUS "Vulkan Packge Lib     : " ${Vulkan_LIBRARIES})
    include_directories(${Vulkan_INCLUDE_DIR})
else(Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan Not Found")
endif(Vulkan_FOUND)

add_subdirectory(source)