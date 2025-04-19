# Define the minimum CMAKE version required:
cmake_minimum_required(VERSION 3.11)

# Define the project:
project(imgui VERSION 3.3.9)

# Set C++ stardard:
set(CMAKE_CXX_STANDARD 11)

# Include OpenGL
# (OpenGL is usually installed by the system per default, so let's find it):
find_package(OpenGL REQUIRED)

# Include GLFW
# (which is in the include-dir, build without examples, tests, etc.):
set(GLFW_DIR ../glfw-3.3/) # Set this to point to an up-to-date GLFW repo
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} glfw_bin EXCLUDE_FROM_ALL)

# Define all source files we want to compile:
set(SOURCES 
    backends/imgui_impl_glfw.cpp
    backends/imgui_impl_opengl3.cpp
    imgui.cpp
    imgui_draw.cpp
    imgui_demo.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
)

# Define library with source files:
add_library(imgui ${SOURCES})

set_target_properties(imgui PROPERTIES FOLDER "imgui")

# IMGUI specific compile definition:
target_compile_definitions(imgui PUBLIC -DImTextureID=ImU64)

# Include directory of glfw:
target_include_directories(imgui PUBLIC ${GLFW_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# Define the libraries to link against:
target_link_libraries(imgui PUBLIC ${OPENGL_LIBRARIES} glfw)
