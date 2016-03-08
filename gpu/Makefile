# Load CUDA using the following command
# module load cuda

#
# Stampede
#
CC = nvcc
MPCC = nvcc
OPENMP = 
CFLAGS = -O3 -arch=sm_35
NVCCFLAGS = -O3 -arch=sm_35
LIBS = -lm

TARGETS = serial gpu

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o GPUframe.o GPUframe.o physics.o
	$(CC) $(NVCCFLAGS) -o $@ $(NVCCLIBS) gpu.o common.o physics.o GPUframe.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu
gpu.o: gpu.cu common.h physics.cuh
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu
physics.o: physics.cu physics.cuh
	$(CC) -c $(NVCCFLAGS) physics.cu
GPUframe.o: GPUframe.cu GPUframe.cuh physics.cuh physics.cu
	$(CC) -c $(NVCCFLAGS) -rdc=true GPUframe.cu physics.cu
	$(CC) -dlink -c $(NVCCFLAGS) GPUframe.o physics.o

clean:
	rm -f *.o $(TARGETS)
