cmake_minimum_required(VERSION 2.6)
project(stitch C CXX)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long")
endif()

get_filename_component(STITCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
set(CMAKE_MODULE_PATH "${STITCH_ROOT}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_PREFIX_PATH "${STITCH_ROOT}" ${CMAKE_PREFIX_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD})

# OpenMP
find_package(OpenMP REQUIRED)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#find_package(OpenMesh REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLFW 3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("../../Libraries/libigl/include") # TODO: Make this legit
include_directories(include)

add_subdirectory(apps)


set(SOURCES
    src/admm_integrator.cpp
    src/backward_euler_integrator.cpp
    src/hinge_energy.cpp
    src/immediate_buckling_energy.cpp
    src/linear_elastic_energy.cpp
    src/nonlinear_elastic_energy.cpp
    src/self_collision_energy.cpp
    src/stitch_energy.cpp
    src/triangle_energies.cpp
)

add_library(stitch STATIC ${SOURCES})


if(MSVC)
	target_compile_definitions(stitch PRIVATE _USE_MATH_DEFINES)
        #target_compile_options(stitch PRIVATE "/bigobj")
endif(MSVC)

#target_link_libraries(stitch ${OPENMESH_CORE_LIBRARY})
