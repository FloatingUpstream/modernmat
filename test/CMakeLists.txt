cmake_minimum_required(VERSION 3.14)

project(modernmatTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)
include(FetchContent)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(modernmat REQUIRED)
  enable_testing()
endif()

if(NOT TARGET GTest::gtest)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(gtest_build_tests OFF CACHE BOOL "" FORCE)
  set(gtest_build_samples OFF CACHE BOOL "" FORCE)
  if(DEFINED CMAKE_CXX_CLANG_TIDY)
    set(_saved_clang_tidy "${CMAKE_CXX_CLANG_TIDY}")
  else()
    unset(_saved_clang_tidy)
  endif()
  set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)
  if(DEFINED _saved_clang_tidy)
    set(CMAKE_CXX_CLANG_TIDY "${_saved_clang_tidy}" CACHE STRING "" FORCE)
  else()
    unset(CMAKE_CXX_CLANG_TIDY CACHE)
  endif()
  foreach(_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
    if(TARGET "${_gtest_target}")
      target_compile_options(
        "${_gtest_target}"
        PRIVATE
          $<$<CXX_COMPILER_ID:MSVC>:/w>
          $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>
      )
    endif()
  endforeach()
endif()

include(GoogleTest)

# ---- Tests ----

add_executable(modernmat_test modernmat_test.cpp)
target_link_libraries(modernmat_test PRIVATE modernmat::modernmat GTest::gtest_main)
target_compile_features(modernmat_test PRIVATE cxx_std_20)

gtest_discover_tests(modernmat_test)

add_executable(
  modernmat_opencv_diff_test
  modernmat_opencv_diff_test.cpp
)
target_link_libraries(
  modernmat_opencv_diff_test
  PRIVATE
    modernmat::modernmat
    GTest::gtest_main
)
target_compile_features(modernmat_opencv_diff_test PRIVATE cxx_std_20)
gtest_discover_tests(modernmat_opencv_diff_test)
add_folders(Differential)

# ---- End-of-file commands ----

add_folders(Test)
