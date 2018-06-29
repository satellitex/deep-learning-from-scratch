add_executable(
        optimizer_test optimizer_test.cpp)
target_link_libraries(optimizer_test
        gtest)

add_test(
        NAME optimizer_test
        COMMAND $<TARGET_FILE:optimizer_test>)
