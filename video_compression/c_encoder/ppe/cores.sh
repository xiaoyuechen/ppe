#!/bin/bash

git checkout master
for b in "openmp" "c+p" "s+p" "c+s+p"
do
  git checkout $b
  for c in 1 2 4
  do
    make clean
    make
    env OMP_NUM_THREADS=$c make perf-stat-record
    echo "========== Branch: $b, cores: $c ==============="
    make perf-stat-report
    echo "================================================"
  done
done

git checkout master
