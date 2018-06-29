add_library(libcurl UNKNOWN IMPORTED)

find_path(libcurl_INCLUDE_DIR libcurl/libcurl.h)
mark_as_advanced(libcurl_INCLUDE_DIR)

find_library(libcurl_LIBRARY libcurl)
mark_as_advanced(libcurl_LIBRARY)

find_library(libcurl_MAIN_LIBRARY libcurl_main)
mark_as_advanced(libcurl_MAIN_LIBRARY)


find_package_handle_standard_args(libcurl DEFAULT_MSG
        libcurl_INCLUDE_DIR
        libcurl_LIBRARY
        libcurl_MAIN_LIBRARY
        )

set(URL https://github.com/curl/curl)
set(VERSION 5d39dde961fc1bed706aff0b84bd9ec24e408a0a)

set_package_properties(libcurl
        PROPERTIES
        URL ${URL}
        DESCRIPTION "Unit test library"
        PURPOSE "commit: ${VERSION}"
        )

if (NOT libcurl_FOUND)
    ExternalProject_Add(curl
            GIT_REPOSITORY ${URL}
            GIT_TAG        ${VERSION}
            CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -Dlibcurl_force_shared_crt=ON
            -Dlibcurl_disable_pthreads=OFF
            INSTALL_COMMAND "" # remove install step
            UPDATE_COMMAND "" # remove update step
            TEST_COMMAND "" # remove test step
            )
    ExternalProject_Get_Property(curl source_dir binary_dir)
    set(libcurl_INCLUDE_DIR ${source_dir}/include)

    set(libcurl_MAIN_LIBRARY ${binary_dir}/lib/liblibcurl_main.a)

    file(MAKE_DIRECTORY ${libcurl_INCLUDE_DIR})

    add_dependencies(libcurl curl)
endif ()

set_target_properties(libcurl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${libcurl_INCLUDE_DIR}
        INTERFACE_LINK_LIBRARIES "pthread;${libcurl_MAIN_LIBRARY}"
        IMPORTED_LOCATION ${libcurl_LIBRARY}
        )
