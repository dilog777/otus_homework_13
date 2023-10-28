project(fashio_mnist VERSION ${PROJECT_VERSION})

file(GLOB_RECURSE SRC *.h *.hpp *.cpp)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE
	fashio_mnist_lib
)

target_include_directories(${PROJECT_NAME} PUBLIC
	${Boost_INCLUDE_DIR}
)
