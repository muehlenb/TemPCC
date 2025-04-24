# Define the minimum CMAKE version required:
cmake_minimum_required(VERSION 3.5)

if (DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "")
endif ()

if (DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
endif ()

option(USE_KINECT "Enable Azure Kinect, HDF5 Stream, etc." ON)

project(TemPCC CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    set(ENV{CCACHE_LOGFILE} "${CMAKE_BINARY_DIR}/ccache.log")
else()
    message(WARNING "ccache not found! The compilation process is carried out without ccache.")
endif()

if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(lib/imgui-1.88/)
add_subdirectory(lib/glad)

find_package(OpenMP)
find_package(CUDAToolkit 12.1 REQUIRED)

# assumes find_package(CUDAToolkit) was already done
if(
    PYTORCH_VERSION VERSION_GREATER_EQUAL 2.5.0 AND
    PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12
)
    message(STATUS "PyTorch NVTX headers workaround: Yes")
    # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        # ensure that PyTorch is told to use NVTX3 headers
        target_compile_definitions(
            CUDA::nvToolsExt INTERFACE
            TORCH_CUDA_USE_NVTX3
        )
        target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
    endif()
else()
    message(STATUS "PyTorch NVTX headers workaround: No")
endif()


find_package(Torch REQUIRED)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS} --extended-lambda")


set(DEFAULT_HEADERS
    # PC Filter
    src/pcfilter/Filter.h

    # PC Renderer
    src/pcrenderer/Renderer.h

    # PC Streamer
    src/pcstreamer/Streamer.h
    src/pcstreamer/BinaryStreamer.h
    src/pcstreamer/binary/BinaryStream.h

    # Point Cloud & other utils
    src/util/OrganizedPointCloud.h
    src/util/Semaphore.h
    src/util/TimeMeasurement.h

    # GL stuff
    src/util/gl/Shader.h
    src/util/gl/Texture2D.h
    src/util/gl/TextureFBO.h
    src/util/gl/GLMesh.h
    src/util/gl/GLRenderable.h

    # Primitives
    src/util/gl/primitive/Triangle.h
    src/util/gl/primitive/Vertex.h
    src/util/gl/primitive/TexCoord.h

    # Coordinate System
    src/util/gl/objects/GLCoordinateSystem.h

    # Math classes
    src/util/math/Vec4.h
    src/util/math/Mat4.h

    # Libtorch Headers
    src/pcrenderer/tempcc/TinyFlowNet.h
    src/pcrenderer/tempcc/TemPCCRenderer.h
    src/pcrenderer/tempcc/TemPCCRendererKernels.h
    src/pcrenderer/tempcc/TemPCCRendererThrust.h
    src/pcrenderer/tempcc/TemPCCStructs.h


    # CUDA Filter:
    src/pcfilter/TemporalNoiseFilter.h
    src/pcfilter/TemporalHoleFiller.h
    src/pcfilter/SpatialHoleFiller.h
    src/pcfilter/ErosionFilter.h
    src/pcfilter/ClippingFilter.h

    # CUDA PDFlow:
    src/pcrenderer/pdflow/PDFlow.h
    src/pcrenderer/pdflow/pdflow_cudalib.h

    # CUDA Hash Set:
    src/util/cuda/CuHashSet.h
    src/util/cuda/CuHashSet.cu

    src/Data.h

    # Precompiled headers for faster compilation
    src/pch.h
)

set(DEFAULT_SOURCES
    # PC Filter:
    src/pcfilter/Filter.cpp

    # PC Renderer:
    src/pcrenderer/Renderer.cpp

    # PC Streamer:
    src/pcstreamer/Streamer.cpp
    src/pcstreamer/binary/BinaryStream.cpp

    # GL stuff
    src/util/gl/Shader.cpp
    src/util/gl/Texture2D.cpp
    src/util/gl/TextureFBO.cpp
    src/util/gl/GLMesh.cpp

    # Coordinate System
    src/util/gl/objects/GLCoordinateSystem.cpp

    # Organized point cloud
    src/util/OrganizedPointCloud.cpp

    # Libtorch Source:
    src/pcrenderer/tempcc/TemPCCRenderer.cpp
    src/pcrenderer/tempcc/TemPCCRendererKernels.cu
    src/pcrenderer/tempcc/TemPCCRendererThrust.cu

    # CUDA Filters:
    src/pcfilter/SpatialHoleFiller.cu
    src/pcfilter/TemporalHoleFiller.cu
    src/pcfilter/TemporalNoiseFilter.cu
    src/pcfilter/ErosionFilter.cu
    src/pcfilter/ClippingFilter.cu

    # PDFlow
    src/pcrenderer/pdflow/pdflow_cudalib.cu

    src/Data.cpp
)

# Define Azure Kinect Files:
if(USE_KINECT)
    set(KINECT_HEADERS

        src/pcstreamer/AzureKinectMKVStreamer.h
        src/pcstreamer/azure_mkv/AzureKinectMKVStream.h
    )

    set(KINECT_SOURCES
    )

    message(STATUS "Azure Kinect Libraries are activated")
endif()

# Define shader & resources which should be listed in IDE:
set(RESOURCES
    # TODO
)

# Allow to include files directly in this paths (without the need to specify folders):
include_directories(src src/util/math lib lib/stb_image lib/tiny_obj_loader)

set(HEADERS ${DEFAULT_HEADERS})
set(SOURCES ${DEFAULT_SOURCES})

if(USE_KINECT)
    list(APPEND HEADERS ${KINECT_HEADERS})
    list(APPEND SOURCES ${KINECT_SOURCES})
endif()

list(APPEND SOURCES src/main.cpp)

# Define executables with source files and resources:
add_executable(TemPCC ${SOURCES} ${HEADERS} ${RESOURCES} src/pch.cpp)

set_property(TARGET TemPCC PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# IMGUI specific compile definition:
target_compile_definitions(TemPCC PUBLIC -DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(TemPCC PUBLIC imgui glad)

target_include_directories(TemPCC PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(TemPCC PRIVATE ${CUDA_LIBRARIES})

target_include_directories(TemPCC PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(TemPCC PRIVATE ${TORCH_LIBRARIES})

if(USE_KINECT)
    # Plattformspezifische Default-Werte setzen
    if(WIN32)
        # Setze nur die Default-Werte, wenn die Benutzervariablen nicht bereits gesetzt wurden
        if("${K4A_INCLUDE_DIR}" STREQUAL "")
            set(K4A_INCLUDE_DIR "C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\include" CACHE PATH "Path to directory which contains k4a and k4arecord includes" FORCE)
        endif()
        if("${K4A_LIB}" STREQUAL "")
            set(K4A_LIB "C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\lib\\k4a.lib" CACHE FILEPATH "Path to k4a.lib" FORCE)
        endif()
        if("${K4A_RECORD_LIB}" STREQUAL "")
            set(K4A_RECORD_LIB "C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\lib\\k4arecord.lib" CACHE FILEPATH "Path to k4arecord.lib" FORCE)
        endif()
    endif()


    target_compile_definitions(TemPCC PRIVATE USE_KINECT)

    # Azure Kinect libraries
    target_include_directories(TemPCC PRIVATE ${K4A_INCLUDE_DIR})
    target_link_libraries(TemPCC PRIVATE ${K4A_LIB})
    target_link_libraries(TemPCC PRIVATE ${K4A_RECORD_LIB})
endif()

target_link_libraries(TemPCC PUBLIC OpenMP::OpenMP_CXX)

# Precompiled Header konfigurieren
target_precompile_headers(TemPCC PRIVATE src/pch.h)
