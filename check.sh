#!/bin/bash
for i in `seq 1 10`;
do
  echo $i
  ./lzss -t -c examples/EXAMPLE$i -o test/COMP$i
  ./ref e examples/EXAMPLE$i test/ref$i
  ./lzss -t -d test/COMP$i -o test/DECOMP$i
  diff examples/EXAMPLE$i test/DECOMP$i
done
