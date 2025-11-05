cmake_minimum_required(VERSION 4.0)
project(neuralneuron)

set(CMAKE_CXX_STANDARD 26)

add_executable(xor xor_main.cpp
        activations.h
        algebra.h
        neuron.h)
