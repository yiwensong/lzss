mkdir zip
cd zip
wget http://mattmahoney.net/dc/enwik8.zip
wget http://mattmahoney.net/dc/enwik9.zip
unzip enwik8.zip
unzip enwik9.zip
mv enwik8 ../examples
mv enwik9 ../examples
cd ..
rm -rf zip
