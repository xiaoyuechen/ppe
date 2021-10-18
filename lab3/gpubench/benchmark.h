/*
 * Copyright (C) 2021  Xiaoyue Chen and Hannah Atmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stddef.h>

void init();
void add_char(size_t size);
void add_float(size_t size);
void load_int_seq(size_t size);
void load_int_rand(size_t size);
void transfer_data(size_t size);

#endif
