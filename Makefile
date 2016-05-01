CC = gcc
NVCC = nvcc
CFLAGS = --std=gnu99 -Ofast -fgcse-sm -fgcse-las
CUFLAGS = # -O0 # -arch=sm_35
LIBS = -lm

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
	./lzss-gpu -t -c examples/EXAMPLE1 -o test/COMPG.EXAMPLE1 > test/COMPG_OUT 2> test/COMPG_ERR
	./lzss-gpu -t -d test/COMPG.EXAMPLE1 -o test/DECOMPG.EXAMPLE1 > test/DCMPG_OUT 2> test/DCMPG_ERR
	./ref e examples/EXAMPLE1 test/ref.EXAMPLE1 > test/REF_OUT 2> test/REF_ERR
	diff examples/EXAMPLE1 test/DECOMPG.EXAMPLE1

check2 : lzss ref
	./lzss -t -c examples/EXAMPLE1 -o test/COMP.EXAMPLE1 > test/COMP_OUT 2> test/COMP_ERR
	./lzss -t -d test/COMP.EXAMPLE1 -o test/DECOMP.EXAMPLE1 > test/DCMP_OUT 2> test/DCMP_ERR
	./ref e examples/EXAMPLE1 test/ref.EXAMPLE1 > test/REF_OUT 2> test/REF_ERR
	diff examples/EXAMPLE1 test/DECOMP.EXAMPLE1

c: check2-gpu check2

clean :
	rm -rf $(TARGETS)
	rm -rf $(OBJECTS)
	rm -rf test/*
	rm -rf *.stackdump
