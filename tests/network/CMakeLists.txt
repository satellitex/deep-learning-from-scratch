add_executable(
        network_test network_test.cpp)
target_link_libraries(network_test
        gtest)

add_test(
        NAME network_test
        COMMAND $<TARGET_FILE:network_test>)
