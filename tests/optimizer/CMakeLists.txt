add_executable(
        optimzier_test optimzier_test.cpp)
target_link_libraries(optimzier_test
        gtest)

add_test(
        NAME optimzier_test
        COMMAND $<TARGET_FILE:optimzier_test>)
