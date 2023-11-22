include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

## OpenACC
find_package(OpenACC REQUIRED)
if (OpenACC_CXX_FOUND)
message("OpenACC FOUND, Version: " ${OpenACC_CXX_VERSION})
endif ()

add_library(simple_ml_ext STATIC
    simple_ml_ext.cpp simple_ml_ext.hpp
)

add_executable(nn
    nn_classifier.cpp 
    simple_ml_ext.hpp)
target_compile_options(nn PRIVATE -O2)
target_link_libraries(nn PRIVATE simple_ml_ext)

add_executable(softmax
    softmax_classifier.cpp 
    simple_ml_ext.hpp)
target_compile_options(softmax PRIVATE -O2)
target_link_libraries(softmax PRIVATE simple_ml_ext)

SET(CMAKE_CXX_COMPILER pgc++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -Minfo=accel")

add_library(simple_ml_openacc STATIC
    simple_ml_openacc.cpp simple_ml_openacc.hpp simple_ml_ext.hpp
)
target_link_libraries(simple_ml_openacc PRIVATE simple_ml_ext)

add_executable(nn_openacc
        nn_classifier_openacc.cpp 
        simple_ml_openacc.hpp)
target_compile_options(nn_openacc PRIVATE -O2)
target_link_libraries(nn_openacc PRIVATE simple_ml_openacc)

add_executable(softmax_openacc
        softmax_classifier_openacc.cpp
        simple_ml_openacc.hpp)
target_compile_options(softmax_openacc PRIVATE -O2)
target_link_libraries(softmax_openacc PRIVATE simple_ml_openacc)

# add_executable(nn
#         nn_classifier.cpp 
#         simple_ml_ext.cpp simple_ml_ext.hpp)
# target_compile_options(nn PRIVATE -O2)

# add_executable(softmax
#         softmax_classifier.cpp 
#         simple_ml_ext.cpp simple_ml_ext.hpp)
# target_compile_options(softmax PRIVATE -O2)

# SET(CMAKE_CXX_COMPILER pgc++)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -Minfo=accel")

# add_executable(nn_openacc
#         nn_classifier_openacc.cpp 
#         simple_ml_openacc.cpp simple_ml_openacc.hpp
#         simple_ml_ext.cpp simple_ml_ext.hpp)
# target_compile_options(nn_openacc PRIVATE -O2)

# add_executable(softmax_openacc
#         softmax_classifier_openacc.cpp
#         simple_ml_openacc.cpp simple_ml_openacc.hpp
#         simple_ml_ext.cpp simple_ml_ext.hpp)
# target_compile_options(softmax_openacc PRIVATE -O2)

