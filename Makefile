CC = cc
NVCC = nvcc
CFLAGS = -O3 --std=gnu99
CUFLAGS = -O3 -arch=sm_35 --std=gnu99
LIBS = -lm

TARGETS = lzss lzss_gpu
OBJECTS = lzss_gpu.ou lzss_gpu_help.ou lzss.o lzss_help.o

all: $(TARGETS)

all: lzss

lzss: lzss_help.o lzss.o
	$(CC) -o $@ $(CFLAGS) $(LIBS) $^

lzss_gpu: lzss_gpu.ou lzss_gpu_help.ou
	$(NVCC) -o $@ $(CFLAGS) $(LIBS) $^

%.o : %.c
	$(CC) -c -o $@ $(CFLAGS) $(LIBS) $<

%.ou : %.cu
	$(NVCC) -c -o $@ $(CFLAGS) $(LIBS) $<

check :		clean lzss
	./lzss

clean :
	rm -rf $(TARGETS)
	rm -rf $(OBJECTS)
