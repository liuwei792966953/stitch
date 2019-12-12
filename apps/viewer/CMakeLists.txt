
include_directories(SYSTEM ${GLFW_INCLUDE_DIR})

include_directories("${STITCH_ROOT}/polyscope/include")
include_directories("${STITCH_ROOT}/polyscope/deps/glm")
include_directories("${STITCH_ROOT}/polyscope/deps/imgui/imgui")
add_subdirectory("${STITCH_ROOT}/polyscope" ${CMAKE_CURRENT_BINARY_DIR}/polyscope)

#include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})
#include_directories(${OPENMESH_INCLUDE_DIR})

include_directories(${STITCH_ROOT}/glad/include)
include_directories(${STITCH_ROOT}/deps/clara/include)
include_directories(${STITCH_ROOT}/src)

set(SRCS
    ${STITCH_ROOT}/glad/src/glad.c
    main.cpp
)

add_executable(viewer ${SRCS})

target_link_libraries(viewer stitch)
target_link_libraries(viewer ${GLFW_LIBRARIES} "-L/usr/local/lib" polyscope)

