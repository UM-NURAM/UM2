um2_add_test(./cross_section.cpp)
um2_add_test(./nuclide.cpp)
um2_add_test(./cross_section_library.cpp)
um2_add_test(./material.cpp)
um2_add_test(./cmfd.cpp)

# Copy the cross section files to the built test directory
file(COPY ${PROJECT_SOURCE_DIR}/tests/physics/xs_files DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

um2_add_test(./mpact_user_xs.cpp)
