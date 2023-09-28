project(test_google VERSION ${PROJECT_VERSION})

file(GLOB_RECURSE SRC *.h *.hpp *.cpp)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE
	GTest::gtest_main
	helloworld_lib
)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
