cmake_minimum_required(VERSION 3.14)

project(video-compression)

add_executable(ppe
  ppe/benchmark_setup.h
  ppe/config.h
  ppe/custom_types.cpp
  ppe/custom_types.h
  ppe/dct8x8_block.cpp
  ppe/dct8x8_block.h
  ppe/gettimeofday.cpp
  ppe/gettimeofday.h
  ppe/main.cpp
  ppe/test_setup.h
  ppe/xml_aux.cpp
  ppe/xml_aux.h)

find_package(libxml2)

target_link_libraries(ppe PRIVATE libxml2)
