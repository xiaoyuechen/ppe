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

#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <time.h>

#define START_TIMER(timer)                                                    \
  struct timeval timer##_start, timer##_end;                                  \
  gettimeofday (&timer##_start, NULL);

#define END_TIMER(timer)                                                      \
  gettimeofday (&timer##_end, NULL);                                          \
  double timer                                                                \
      = ((double)(timer##_end.tv_sec) - (double)(timer##_start.tv_sec))       \
        + ((double)(timer##_end.tv_usec) - (double)(timer##_start.tv_usec))   \
              / 1E6;

#endif
