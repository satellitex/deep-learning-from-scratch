SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/test_bin)

# Reusable tests
add_subdirectory(primitive)
add_subdirectory(layer)
add_subdirectory(network)
add_subdirectory(optimizer)
add_subdirectory(loader)