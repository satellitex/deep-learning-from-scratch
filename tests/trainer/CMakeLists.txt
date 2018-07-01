add_executable(
        trainer_test trainer_test.cpp)
target_link_libraries(trainer_test
        gtest)

add_test(
        NAME trainer_test
        COMMAND $<TARGET_FILE:trainer_test>)
