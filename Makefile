CC = gcc
NVCC = nvcc
CFLAGS = --std=gnu99 -Ofast -fgcse-sm -fgcse-las
CUFLAGS = -Ofast -arch=sm_35 --std=gnu99
LIBS = -lm

TARGETS = lzss ref # lzss_gpu
OBJECTS = lzss_gpu.ou lzss_gpu_help.ou common.ou lzss.o lzss_help.o common.o ref.o

all: $(TARGETS)

lzss: lzss_help.o lzss.o common.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

ref: ref.o common.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

lzss_gpu: lzss_gpu.ou lzss_gpu_help.ou common.ou
	$(NVCC) -o $@ $(CFLAGS) $(LIBS) $^

%.o : %.c
	$(CC) -c -o $@ $(CFLAGS) $(LIBS) $<

%.ou : %.cu
	$(NVCC) -c -o $@ $(CFLAGS) $(LIBS) $<

check :		clean lzss ref
	./check.sh

check2 : clean lzss
	./lzss -t -c examples/EXAMPLE1 -o test/COMP.EXAMPLE1 > test/COMP_OUT 2> test/COMP_ERR
	./lzss -t -d test/COMP.EXAMPLE1 -o test/DECOMP.EXAMPLE1 > test/DCMP_OUT 2> test/DCMP_ERR
	diff examples/EXAMPLE1 test/DECOMP.EXAMPLE1

clean :
	rm -rf $(TARGETS)
	rm -rf $(OBJECTS)
	rm -rf test/*
	rm -rf *.stackdump
