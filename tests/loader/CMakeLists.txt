add_executable(
        loader_test loader_test.cpp)
target_link_libraries(loader_test
        gtest)

add_test(
        NAME loader_test
        COMMAND $<TARGET_FILE:loader_test>)
