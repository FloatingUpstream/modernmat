cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    modernmat
    VERSION 0.1.0
    DESCRIPTION "Header-only copy-on-write facade for OpenCV cv::Mat"
    HOMEPAGE_URL "https://github.com/modernmat/modernmat"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ---- Declare library ----

add_library(modernmat_modernmat INTERFACE)
add_library(modernmat::modernmat ALIAS modernmat_modernmat)

set_property(
    TARGET modernmat_modernmat PROPERTY
    EXPORT_NAME modernmat
)

target_include_directories(
    modernmat_modernmat ${warning_guard}
    INTERFACE
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "\$<INSTALL_INTERFACE:include>"
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(modernmat_modernmat INTERFACE cxx_std_20)
target_link_libraries(
    modernmat_modernmat
    INTERFACE
      ${OpenCV_LIBS}
)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Examples ----

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${modernmat_DEVELOPER_MODE}")
  if(BUILD_EXAMPLES)
add_subdirectory(example)
  endif()
endif()

# ---- Developer mode ----

if(NOT modernmat_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of modernmat"
  )
endif()

include(cmake/dev-mode.cmake)
