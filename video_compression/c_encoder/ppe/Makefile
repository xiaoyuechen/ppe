#!/usr/bin/make -f

# Makefile for ppe
# Copyright (C) 2021  Xiaoyue Chen and Hannah Atmer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

SHELL = /bin/sh

EXEC = cencoder

ARGS =

PERF_STAT_RECORD_FILE = perf-stat.data
PERF_RECORD_EVENTS = task-clock,cycles,instructions,cache-misses,branch-misses
PERF_RECORD_FILE = perf-record.data
PERF_CALL_GRAPH_FILE = $(EXEC)-callgraph.png

CXX_SRCS = custom_types.cpp dct8x8_block.cpp main.cpp xml_aux.cpp
C_SRCS = cmd_args.c opt_opencl.c opt_openacc.c
OBJS = $(CXX_SRCS:.cpp=.o) $(C_SRCS:.c=.o)

DEBUG_FLAGS = -g

CC = gcc
CXX = g++

CPPFLAGS = $(DEBUG_FLAGS) -O0 -I /usr/include/libxml2/ -fopenmp \
	-msse4.1 -mavx2 -fopenacc
CFLAGS = -std=c99
CXXFLAGS = -std=c++17
LDFLAGS = $(PERF_FLAGS)
LDLIBS = -lxml2 -ltiff -fopenmp -lOpenCL

.PHONY: all
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

custom_types.o: custom_types.h config.h
dct8x8_block.o: dct8x8_block.h
xml_aux.o: xml_aux.h config.h
cmd_args.o: cmd_args.h
opt_opencl.o: opt_opencl.h kernel.cl
opt_openacc.o: opt_openacc.h
main.o: config.h custom_types.h dct8x8_block.h xml_aux.h cmd_args.h \
	opt_opencl.h opt_openacc.h

.PHONY: clean
clean:
	rm $(EXEC) $(OBJS)

.PHONY: run
run:
	./$(EXEC) $(ARGS)

.PHONY: test
test: run
	matlab -sd ../../ml_encoder -batch compare_encoders

.PHONY: perf-stat-record
perf-stat-record:
	perf stat record -o $(PERF_STAT_RECORD_FILE) ./$(EXEC)

.PHONY: perf-stat-report
perf-stat-report:
	perf stat report -i $(PERF_STAT_RECORD_FILE)

.PHONY: perf-record
perf-record:
	perf record -e $(PERF_RECORD_EVENTS) -g -o $(PERF_RECORD_FILE) ./$(EXEC)

.PHONY: perf-report
perf-report:
	perf report --no-children -i $(PERF_RECORD_FILE)

.PHONY: perf-call-graph
perf-call-graph:
	perf script -i $(PERF_RECORD_FILE) | c++filt | gprof2dot -n 4 -f perf | dot -Tpng -o \
	$(PERF_CALL_GRAPH_FILE)
