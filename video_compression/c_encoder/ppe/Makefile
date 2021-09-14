#!/usr/bin/make -f

SHELL = /bin/sh
EXEC = ppe
SRCS = custom_types.cpp dct8x8_block.cpp main.cpp xml_aux.cpp
OBJS = $(SRCS:.cpp=.o)

CXX = g++
CXXDEBUG = -g
CXXFLAGS = $(CXXDEBUG) -I /usr/include/libxml2/ -fopenmp -msse4.1 -std=c++17
LDFLAGS = -g
LDLIBS = -lxml2 -ltiff -fopenmp

.PHONY: all
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

custom_types.o: custom_types.h config.h
dct8x8_block.o: dct8x8_block.h
xml_aux.o: xml_aux.h config.h
main.o: config.h custom_types.h dct8x8_block.h xml_aux.h

.PHONY: clean
clean:
	rm $(EXEC) $(OBJS)
