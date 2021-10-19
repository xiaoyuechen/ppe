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

#include <stdlib.h>

#include "benchmark.h"

#define SIZE 1 * 1000 * 1000 * 1000ULL

int
main (int argc, char *argv[argc + 1])
{
  init ();
  add_char (SIZE);
  add_float (SIZE);
  load_int_seq (SIZE);
  load_int_rand (SIZE);
  transfer_data (SIZE);

  exit (EXIT_SUCCESS);
}
