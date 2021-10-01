#!/bin/bash
HEIGHT=1024
WIDTH=1024
PIXELS=$HEIGHT*$WIDTH

for c in $((PIXELS/16)) $((PIXELS/32)) $((PIXELS/64)) $((PIXELS/128)) $((PIXELS/256)) $((PIXELS/512)) $((PIXELS/1024)) 
do
  echo "========== Work per GPU Core: $c ==============="
  ./cencoder -c $c -cl
  echo "================================================"
done