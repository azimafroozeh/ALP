add_executable(test_alp_sample test_alp.cpp)
target_link_libraries(test_alp_sample PRIVATE gtest_main ALP)
gtest_discover_tests(test_alp_sample)


