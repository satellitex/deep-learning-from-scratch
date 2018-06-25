add_executable(
        layer_test layer_test.cpp)
target_link_libraries(layer_test
        gtest)

add_test(
        NAME layer_test
        COMMAND $<TARGET_FILE:layer_test>)
