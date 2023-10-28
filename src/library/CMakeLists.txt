project(fashio_mnist_lib VERSION ${PROJECT_VERSION})

file(GLOB_RECURSE SRC *.h *.hpp *.cpp)

add_library(${PROJECT_NAME} STATIC ${SRC})

target_include_directories(${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${tensorflow_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
	${tensorflow_SOURCE_DIR}/lib/tensorflow.lib
)
