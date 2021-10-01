#!/bin/bash

for num_thd in 16 32 64 128 256 512 1024
do
  echo "========== Work per GPU Core: $num_thd ==============="
  ./cencoder -c --cl_num_thd=$num_thd > outputs
  cat ../../outputs/execution_stats.txt
  echo "================================================"
done
