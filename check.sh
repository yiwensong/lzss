#!/bin/bash
if [ ! -d "./test" ];
then
  mkdir -p ./test;
fi;

for i in $( /usr/bin/ls examples );
do
  echo -e '\n\n\n'
  echo $i
  echo -e ''
  ./lzss -c examples/$i -o test/COMP.$i
  ./lzss -t -c examples/$i -o test/COMP.$i
  ./ref e examples/$i test/ref.$i
  ./lzss -t -d test/COMP.$i -o test/DECOMP.$i
  diff examples/$i test/DECOMP.$i
done
