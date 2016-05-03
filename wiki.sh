mkdir unzip
wget http://mattmahoney.net/dc/enwik9.zip unzip/WIKI9.zip
wget http://mattmahoney.net/dc/enwik8.zip unzip/WIKI8.zip
unzip unzip/WIKI9.zip examples/WIKI9
unzip unzip/WIKI8.zip examples/WIKI8
rm -rf unzip
