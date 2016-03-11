# Load CUDA using the following command
# module load cuda

#
# Stampede
#
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

all:    $(TARGETS)

serial: serial.o common.o
        $(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o frame_utils.o GPUframe.o physics.o
        $(CC) $(NVCCFLAGS) -o $@ $(NVCCLIBS) gpu.o common.o physics.o frame_utils.o GPUframe.o

serial.o: serial.cu common.h
        $(CC) -c $(CFLAGS) serial.cu
gpu.o: gpu.cu common.h physics.cuh
        $(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
        $(CC) -c $(CFLAGS) common.cu
physics.o: physics.cu physics.cuh
        $(CC) -c $(NVCCFLAGS) physics.cu
frame_utils.o: frame_utils.cu frame_utils.cuh
        $(CC) -c $(NVCCFLAGS) frame_utils.cu
GPUframe.o: GPUframe.cu GPUframe.cuh
        $(CC) -c $(NVCCFLAGS) GPUframe.cu

clean:
        rm -f *.o $(TARGETS)
