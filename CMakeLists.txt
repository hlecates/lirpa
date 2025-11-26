cmake_minimum_required(VERSION 3.16)
project(LIRPA)

set(LIRPA_VERSION 1.0.0)
add_definitions("-DLIRPA_VERSION=\"${LIRPA_VERSION}\"")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##################
## User options ##
##################

option(BUILD_STATIC_LIRPA "Build static LIRPA binary" OFF)
option(ENABLE_OPENBLAS "Do symbolic bound tightening using blas" ON)

###################
## Git variables ##
###################

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
add_definitions("-DGIT_BRANCH=\"${GIT_BRANCH}\"")

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
add_definitions("-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

#########################
## Basic configuration ##
#########################

set(TOOLS_DIR "${PROJECT_SOURCE_DIR}/tools")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(RESOURCES_DIR "${PROJECT_SOURCE_DIR}/resources")
set(COMMON_DIR "${SRC_DIR}/common")
set(NLR_DIR "${SRC_DIR}/nlr")
set(INPUT_PARSERS_DIR "${SRC_DIR}/input_parsers")

if (MSVC)
  set(SCRIPT_EXTENSION bat)
else()
  set(SCRIPT_EXTENSION sh)
endif()

###########
## Boost ##
###########

add_definitions(-DBOOST_NO_CXX98_FUNCTION_BASE)
set(BOOST_VERSION 1.84.0)
set(BOOST_DIR "${TOOLS_DIR}/boost-${BOOST_VERSION}")

if (MSVC)
  set(BOOST_ROOT "${BOOST_DIR}/win_installed")
  set(Boost_NAMESPACE libboost)
elseif (${CMAKE_SIZEOF_VOID_P} EQUAL 4 AND NOT MSVC)
  set(BOOST_ROOT "${BOOST_DIR}/installed32")
else()
  set(BOOST_ROOT "${BOOST_DIR}/installed")
endif()

set(Boost_USE_DEBUG_RUNTIME FALSE)
find_package(Boost ${BOOST_VERSION} COMPONENTS program_options timer chrono thread regex)

if (NOT ${Boost_FOUND})
  execute_process(COMMAND ${TOOLS_DIR}/download_boost.${SCRIPT_EXTENSION} ${BOOST_VERSION})
  find_package(Boost ${BOOST_VERSION} REQUIRED COMPONENTS program_options timer chrono thread regex)
endif()

set(LIBS_INCLUDES ${Boost_INCLUDE_DIRS})
list(APPEND LIBS ${Boost_LIBRARIES})

##############
## Protobuf ##
##############

set(PROTOBUF_VERSION 3.19.2)
set(PROTOBUF_DEFAULT_DIR "${TOOLS_DIR}/protobuf-${PROTOBUF_VERSION}")

if (NOT PROTOBUF_DIR)
    set(PROTOBUF_DIR ${PROTOBUF_DEFAULT_DIR})
endif()

if(NOT EXISTS "${PROTOBUF_DIR}/installed/lib/libprotobuf.a")
    message("Can't find protobuf, installing...")
    if (${PROTOBUF_DIR} STREQUAL ${PROTOBUF_DEFAULT_DIR})
        execute_process(COMMAND ${TOOLS_DIR}/download_protobuf.sh ${PROTOBUF_VERSION})
    else()
        message(FATAL_ERROR "Can't find protobuf in the supplied directory")
    endif()
endif()

set(PROTOBUF_LIB protobuf)
add_library(${PROTOBUF_LIB} SHARED IMPORTED)
set_property(TARGET ${PROTOBUF_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROTOBUF_LIB} PROPERTIES IMPORTED_LOCATION ${PROTOBUF_DIR}/installed/lib/libprotobuf.a)
target_include_directories(${PROTOBUF_LIB} INTERFACE ${PROTOBUF_DIR}/installed/include)
list(APPEND LIBS ${PROTOBUF_LIB})

##########
## ONNX ##
##########

set(ONNX_VERSION 1.15.0)
set(ONNX_DIR "${TOOLS_DIR}/onnx-${ONNX_VERSION}")

if(NOT EXISTS "${ONNX_DIR}/onnx.proto3.pb.h")
    message("Generating ONNX protobuf file...")
    execute_process(COMMAND ${TOOLS_DIR}/download_onnx.sh ${ONNX_VERSION} ${PROTOBUF_VERSION})
endif()

file(GLOB DEPS_ONNX "${ONNX_DIR}/*.cc")
include_directories(SYSTEM ${ONNX_DIR})

#############
## PyTorch ##
#############

set(PYTORCH_VERSION 2.2.1)
set(PYTORCH_DIR "${TOOLS_DIR}/libtorch-${PYTORCH_VERSION}")
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_DIR})

if(NOT EXISTS "${PYTORCH_DIR}")
    execute_process(COMMAND ${TOOLS_DIR}/download_libtorch.sh ${PYTORCH_VERSION})
endif()

find_package(Torch ${PYTORCH_VERSION} REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
list(APPEND LIBS ${TORCH_LIBRARIES})

##############
## OpenBLAS ##
##############

if (NOT MSVC AND ${ENABLE_OPENBLAS})
  set(OPENBLAS_VERSION 0.3.19)
  set(OPENBLAS_LIB openblas)
  set(OPENBLAS_DEFAULT_DIR "${TOOLS_DIR}/OpenBLAS-${OPENBLAS_VERSION}")

  if (NOT OPENBLAS_DIR)
      set(OPENBLAS_DIR ${OPENBLAS_DEFAULT_DIR})
  endif()

  message(STATUS "Using OpenBLAS for matrix multiplication")
  add_compile_definitions(ENABLE_OPENBLAS)

  if(NOT EXISTS "${OPENBLAS_DIR}/installed/lib/libopenblas.a")
    message("Can't find OpenBLAS, installing...")
    if (${OPENBLAS_DIR} STREQUAL ${OPENBLAS_DEFAULT_DIR})
      execute_process(COMMAND ${TOOLS_DIR}/download_openBLAS.sh ${OPENBLAS_VERSION})
    else()
      message(FATAL_ERROR "Can't find OpenBLAS in the supplied directory")
    endif()
  endif()

  add_library(${OPENBLAS_LIB} SHARED IMPORTED)
  set_target_properties(${OPENBLAS_LIB} PROPERTIES IMPORTED_LOCATION ${OPENBLAS_DIR}/installed/lib/libopenblas.a)
  list(APPEND LIBS ${OPENBLAS_LIB})
  target_include_directories(${OPENBLAS_LIB} INTERFACE ${OPENBLAS_DIR}/installed/include)
endif()

###########
## Build ##
###########

set(LIRPA_LIB LIRPACore)
set(BIN_DIR "${CMAKE_BINARY_DIR}/bin")

# Collect source files
file(GLOB SRCS_NLR "${NLR_DIR}/*.cpp")
file(GLOB SRCS_NLR_CONV "${NLR_DIR}/conv/*.cpp")
file(GLOB SRCS_INPUT_PARSERS "${INPUT_PARSERS_DIR}/*.cpp")
file(GLOB SRCS_COMMON "${COMMON_DIR}/*.cpp")
file(GLOB SRCS_COMMON_REAL "${COMMON_DIR}/real/*.cpp")
# LIRPA: Options.cpp not needed - exclude configuration directory from build
# file(GLOB SRCS_CONFIG "${SRC_DIR}/configuration/*.cpp")
set(SRCS_CONFIG "${SRC_DIR}/configuration/GlobalConfiguration.cpp")

# Build core library
add_library(${LIRPA_LIB}
    ${DEPS_ONNX}
    ${SRCS_NLR}
    ${SRCS_NLR_CONV}
    ${SRCS_INPUT_PARSERS}
    ${SRCS_COMMON}
    ${SRCS_COMMON_REAL}
    ${SRCS_CONFIG}
)
target_include_directories(${LIRPA_LIB} PUBLIC
    ${SRC_DIR}
    ${COMMON_DIR}
    ${NLR_DIR}
    ${INPUT_PARSERS_DIR}
    ${SRC_DIR}/engine
    ${SRC_DIR}/configuration
    ${LIBS_INCLUDES}
)

# Compiler flags
if (NOT MSVC)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(COMPILE_FLAGS -Wall -Wextra -MMD -Qunused-arguments -Wno-deprecated-declarations -Wno-unused-but-set-variable)
    elseif (CMAKE_BUILD_TYPE MATCHES "Release")
        set(COMPILE_FLAGS -Wall)
    else()
        set(COMPILE_FLAGS -Wall -Wextra -MMD)
    endif()
    set(RELEASE_FLAGS ${COMPILE_FLAGS} -O3)
endif()

add_definitions(-DRESOURCES_DIR="${RESOURCES_DIR}")

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND LIBS Threads::Threads)

# Link libraries
if (BUILD_STATIC_LIRPA)
  target_link_libraries(${LIRPA_LIB} ${LIBS} -static)
else()
  target_link_libraries(${LIRPA_LIB} ${LIBS})
endif()

target_compile_options(${LIRPA_LIB} PRIVATE ${RELEASE_FLAGS})

# Build executables
add_executable(RunCROWN RunCROWN.cpp)
target_link_libraries(RunCROWN ${LIRPA_LIB})
target_include_directories(RunCROWN PRIVATE ${LIBS_INCLUDES})
target_compile_options(RunCROWN PRIVATE ${RELEASE_FLAGS})

add_executable(RunAlphaCROWN RunAlphaCROWN.cpp)
target_link_libraries(RunAlphaCROWN ${LIRPA_LIB})
target_include_directories(RunAlphaCROWN PRIVATE ${LIBS_INCLUDES})
target_compile_options(RunAlphaCROWN PRIVATE ${RELEASE_FLAGS})

add_executable(RunFull RunFull.cpp)
target_link_libraries(RunFull ${LIRPA_LIB})
target_include_directories(RunFull PRIVATE ${LIBS_INCLUDES})
target_compile_options(RunFull PRIVATE ${RELEASE_FLAGS})

add_executable(TestFlatten TestFlatten.cpp)
target_link_libraries(TestFlatten ${LIRPA_LIB})
target_include_directories(TestFlatten PRIVATE ${LIBS_INCLUDES})
target_compile_options(TestFlatten PRIVATE ${RELEASE_FLAGS})

# Copy executables to bin directory
add_custom_command(TARGET RunCROWN POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:RunCROWN> ${BIN_DIR}/)

add_custom_command(TARGET RunAlphaCROWN POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:RunAlphaCROWN> ${BIN_DIR}/)

add_custom_command(TARGET RunFull POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:RunFull> ${BIN_DIR}/)

message(STATUS "LIRPA configured successfully")
message(STATUS "  - LIRPA Version: ${LIRPA_VERSION}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - PyTorch version: ${PYTORCH_VERSION}")
message(STATUS "  - Boost version: ${BOOST_VERSION}")
