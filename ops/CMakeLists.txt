set(GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# find torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Make as shared library
add_library(image_laplacian SHARED image_laplacian.cpp)
# Enable C++14
target_compile_features(image_laplacian PRIVATE cxx_std_14)
set_property(TARGET image_laplacian PROPERTY CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(image_laplacian "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(image_laplacian opencv_core opencv_imgproc)