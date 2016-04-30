#!/bin/bash
if [ ! -d "./test" ];
then
  mkdir -p ./test;
fi;

l=$(which ls)

for i in $( $l examples );
do
  echo -e '\n\n\n'
  echo $i
  echo -e ''
  ./lzss-gpu -c examples/$i -o test/COMPG.$i
  ./lzss-gpu -t -c examples/$i -o test/COMPG.$i
  ./ref e examples/$i test/ref.$i
  ./lzss-gpu -t -d test/COMPG.$i -o test/DECOMPG.$i
  diff examples/$i test/DECOMPG.$i
done
