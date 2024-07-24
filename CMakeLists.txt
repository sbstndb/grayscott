# Minimum version de CMake requise
cmake_minimum_required(VERSION 3.10)

# Nom du projet
project(GrayScott)

# Spécifier la version du standard C++ (ici C++11)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ajouter l'option pour OpenMP
find_package(OpenMP REQUIRED)

# Ajouter l'exécutable
add_executable(grayscott src/main.cpp)

# Lier OpenMP à l'exécutable
target_link_libraries(grayscott PUBLIC OpenMP::OpenMP_CXX)

