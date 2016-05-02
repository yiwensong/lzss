CC = gcc
NVCC = nvcc
CFLAGS = --std=gnu99 -Ofast -fgcse-sm -fgcse-las
CUFLAGS = # -O0 # -arch=sm_35
LIBS = -lm
e = 2

TARGETS = lzss ref lzss-gpu
OBJECTS = lzss_gpu.ou lzss_gpu_help.ou common.ou lzss.o lzss_help.o common.o ref.o

all: $(TARGETS)

lzss: lzss_help.o lzss.o common.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

ref: ref.o common.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

lzss-gpu: lzss-gpu.cu lzss_gpu_help.cu common.cu
	$(NVCC) -o $@ $(CUFLAGS) $(LIBS) $^

gpu: lzss-gpu

%.o : %.c
	$(CC) -c -o $@ $(CFLAGS) $(LIBS) $<

check :		clean lzss ref
	./check.sh

check-gpu : clean gpu ref
	./check-gpu.sh

check2-gpu : gpu ref
	./lzss-gpu -t -c examples/EXAMPLE$e -o test/COMPG.EXAMPLE$e > test/COMPG_OUT 2> test/COMPG_ERR
	./lzss-gpu -t -d test/COMPG.EXAMPLE$e -o test/DECOMPG.EXAMPLE$e > test/DCMPG_OUT 2> test/DCMPG_ERR
	./ref e examples/EXAMPLE$e test/ref.EXAMPLE$e > test/REF_OUT 2> test/REF_ERR
	diff examples/EXAMPLE$e test/DECOMPG.EXAMPLE$e

check2 : lzss ref
	./lzss -t -c examples/EXAMPLE$e -o test/COMP.EXAMPLE$e > test/COMP_OUT 2> test/COMP_ERR
	./lzss -t -d test/COMP.EXAMPLE$e -o test/DECOMP.EXAMPLE$e > test/DCMP_OUT 2> test/DCMP_ERR
	./ref e examples/EXAMPLE$e test/ref.EXAMPLE$e > test/REF_OUT 2> test/REF_ERR
	diff examples/EXAMPLE$e test/DECOMP.EXAMPLE$e

c: check2-gpu check2

clean :
	rm -rf $(TARGETS)
	rm -rf $(OBJECTS)
	rm -rf test/*
	rm -rf *.stackdump
