macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# CMakeList.txt: проект CMake для knn_ocr_2; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.19 FATAL_ERROR)

# Добавляем в проект CUDA и обозначаем стандарт
project ("knn_ocr_2" LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# здесь любая валидная сборка OpenCV идентичная моложе 4.5.3
# у меня просто их несколько
find_package( OpenCV REQUIRED EXACT 4.5.3 PATHS "C:/opencv/build" )
# find_package(Boost 1.78.0 COMPONENTS filesystem) в C++17 есть бусто-заменитель
find_package(CUDAToolkit)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDA_INCLUDE_DIRS} )

# Обьявим CUDA-код как библиотеку и добавим его в проект
add_library(knn STATIC "knn.cu" "knn.cuh")
target_compile_features(knn PUBLIC cxx_std_17)
set_target_properties(knn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Только для MSVC (вернее, для всех компилеров, но понимает это только MSVC :) )
add_compile_options("/Zc:__cplusplus")
add_compile_options("/openmp")

# Добавьте источник в исполняемый файл этого проекта.
add_executable (knn_ocr_2 "knn_ocr_2.cpp" "knn_ocr_2.h")
# add_executable (knn_ocr_2 "knn.cu" "knn.cuh")
set_target_properties(knn PROPERTIES CUDA_ARCHITECTURES "50")
target_link_libraries( knn CUDA::cudart )
target_link_libraries( knn_ocr_2 ${OpenCV_LIBS} knn)
target_compile_options( knn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>) # поддерживать openmp в файлах cu (nvcc)
print_all_variables()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
