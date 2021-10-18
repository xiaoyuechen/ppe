#!/usr/bin/make -f

# Makefile for microbenchmark
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
EXEC = CPUBenchmarks

SRCS = benchmarks.c
OBJS = $(SRCS:.c=.o)

DEBUG_FLAGS = -g
CPPFLAGS = $(DEBUG_FLAGS) -O1 -I /usr/include/libxml2/ -fopenmp -mavx
	#-msse4.2 -mavx2 -fopenacc
CXXFLAGS = -std=c++17

.PHONY: all
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)
