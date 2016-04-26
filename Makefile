CC = gcc
NVCC = nvcc
CFLAGS = --std=gnu99 -Ofast -fgcse-sm -fgcse-las
CUFLAGS = #-Ofast -arch=sm_35 --std=gnu99
LIBS = -lm

TARGETS = lzss ref # lzss_gpu
OBJECTS = lzss_gpu.ou lzss_gpu_help.ou common.ou lzss.o lzss_help.o common.o ref.o

all: $(TARGETS)

lzss: lzss_help.o lzss.o common.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

ref: ref.o common.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

lzss-gpu: lzss-gpu.cu lzss_gpu_help.cu common.cu
	$(NVCC) -o $@ $(CUFLAGS) $(LIBS) $^

%.o : %.c
	$(CC) -c -o $@ $(CFLAGS) $(LIBS) $<

%.ou : %.cu
	$(NVCC) -c -o $@ $(CUFLAGS) $(LIBS) $<

check :		clean lzss ref
	./check.sh

check2 : clean lzss ref
	./lzss -t -c examples/EXAMPLE2 -o test/COMP.EXAMPLE2 > test/COMP_OUT 2> test/COMP_ERR
	./lzss -t -d test/COMP.EXAMPLE2 -o test/DECOMP.EXAMPLE2 > test/DCMP_OUT 2> test/DCMP_ERR
	./ref e examples/EXAMPLE2 test/ref.EXAMPLE2 > test/REF_OUT 2> test/REF_ERR
	diff examples/EXAMPLE2 test/DECOMP.EXAMPLE2

clean :
	rm -rf $(TARGETS)
	rm -rf $(OBJECTS)
	rm -rf test/*
	rm -rf *.stackdump
