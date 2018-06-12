add_library(eigen UNKNOWN IMPORTED)

find_path(eigen_INCLUDE_DIR eigen/eigen.h)
mark_as_advanced(eigen_INCLUDE_DIR)

find_library(eigen_LIBRARY eigen)
mark_as_advanced(eigen_LIBRARY)

find_library(eigen_MAIN_LIBRARY eigen_main)
mark_as_advanced(eigen_MAIN_LIBRARY)

find_package_handle_standard_args(eigen DEFAULT_MSG
        eigen_INCLUDE_DIR
        eigen_LIBRARY
        eigen_MAIN_LIBRARY
        )

set(URL https://github.com/eigenteam/eigen-git-mirror)
set(VERSION 3.3.4)

set_package_properties(eigen
        PROPERTIES
        URL ${URL}
        DESCRIPTION "Eigen library"
        PURPOSE "commit: ${VERSION}"
        )

if (NOT eigen_FOUND)
    ExternalProject_Add(eigen_lib
            GIT_REPOSITORY ${URL}
            GIT_TAG        ${VERSION}
            INSTALL_COMMAND "" # remove install step
            UPDATE_COMMAND "" # remove update step
            TEST_COMMAND "" # remove test step
            )
    ExternalProject_Get_Property(eigen_lib source_dir binary_dir)
    include_directories(
            ${source_dir}
    )
endif ()
