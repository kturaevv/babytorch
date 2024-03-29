# Cmake file to print out source and test files on separate lines
if (DEFINED SOURCE_FILES)
    message(STATUS "Source Files:")
    foreach(SOURCE_FILE IN LISTS SOURCE_FILES)
        message(STATUS "  ${SOURCE_FILE}")
    endforeach()
else()
    message(WARNING "SOURCE_FILES variable is not set.")
endif()

if (DEFINED TEST_FILES)
    message(STATUS "Test Files:")
    foreach(TEST_FILE IN LISTS TEST_FILES)
        message(STATUS "  ${TEST_FILE}")
    endforeach()
else()
    message(WARNING "TEST_FILES variable is not set.")
endif()

