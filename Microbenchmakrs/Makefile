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
EXEC = microbenchmark

CXX_SRCS = add_benchmark.cpp
OBJS = $(CXX_SRCS:.cpp=.o) $(C_SRCS:.c=.o)

DEBUG_FLAGS = -g
CPPFLAGS = $(DEBUG_FLAGS) -O0 -I /usr/include/libxml2/ -fopenmp \
	-msse4.1 -mavx2 -fopenacc
CFLAGS = -std=c99
CXXFLAGS = -std=c++17

.PHONY: all
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)
