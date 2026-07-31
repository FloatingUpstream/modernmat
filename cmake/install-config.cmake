include(CMakeFindDependencyMacro)
find_dependency(OpenCV COMPONENTS core imgproc)

include("${CMAKE_CURRENT_LIST_DIR}/modernmatTargets.cmake")
