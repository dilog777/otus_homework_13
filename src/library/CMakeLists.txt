project(fashio_mnist_lib VERSION ${PROJECT_VERSION})

file(GLOB_RECURSE SRC *.h *.hpp *.cpp)

add_library(${PROJECT_NAME} STATIC ${SRC})

target_include_directories(${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${tensorflow_SOURCE_DIR}/include
)

find_library(tensorflow_lib tensorflow PATHS ${tensorflow_SOURCE_DIR}/lib/)

target_link_libraries(${PROJECT_NAME} PRIVATE 
	${tensorflow_lib}
)
