add_library(opencv UNKNOWN IMPORTED)

find_path(opencv_INCLUDE_DIR opencv/opencv.h)
mark_as_advanced(opencv_INCLUDE_DIR)

find_library(opencv_LIBRARY opencv)
mark_as_advanced(opencv_LIBRARY)

find_library(opencv_MAIN_LIBRARY opencv_main)
mark_as_advanced(opencv_MAIN_LIBRARY)

find_package_handle_standard_args(opencv DEFAULT_MSG
        opencv_INCLUDE_DIR
        opencv_LIBRARY
        opencv_MAIN_LIBRARY
        )

set(URL https://github.com/opencv/opencv)
set(VERSION 3.4.1)

set_package_properties(opencv
        PROPERTIES
        URL ${URL}
        DESCRIPTION "opencv library"
        PURPOSE "commit: ${VERSION}"
        )

if (NOT opencv_FOUND)
    ExternalProject_Add(opencv_lib
            GIT_REPOSITORY ${URL}
            GIT_TAG        ${VERSION}
            CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DBUILD_DOCS:BOOL=FALSE
                -DBUILD_EXAMPLES:BOOL=FALSE
                -DBUILD_TESTS:BOOL=FALSE
                -DBUILD_SHARED_LIBS:BOOL=TRUE
                -DWITH_CUDA:BOOL=FALSE
                -DWITH_FFMPEG:BOOL=FALSE
                -DBUILD_PERF_TESTS:BOOL=FALSE
            BUILD_BYPRODUCTS ${EP_PREFIX}/src/opencv-build/opencv2/opencv/libopencv_main.a
            ${EP_PREFIX}/src/opencv-build/opencv2/opencv/libopencv.a
            ${EP_PREFIX}/src/opencv-build/opencv2/libgmock_main.a
            ${EP_PREFIX}/src/opencv-build/opencv2/libgmock.a
            INSTALL_COMMAND "" # remove install step
            UPDATE_COMMAND "" # remove update step
            TEST_COMMAND "" # remove test step
            )

    ExternalProject_Get_Property(opencv_lib source_dir binary_dir)

    set(opencv_INCLUDE_DIR "${source_dir}/modules/core/include;${source_dir}/include;${binary_dir}")
    set(opencv_MAIN_LIBRARY ${binary_dir}/lib/libopencv_core.dylib)
    set(opencv_LIBRARY ${binary_dir}/lib/libopencv_core.dylib)

    file(MAKE_DIRECTORY ${opencv_INCLUDE_DIR})

    add_dependencies(opencv opencv_lib)
endif ()

set_target_properties(opencv PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${opencv_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "pthread;${opencv_MAIN_LIBRARY}"
        IMPORTED_LOCATION ${opencv_LIBRARY}
        )
