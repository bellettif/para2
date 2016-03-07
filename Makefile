#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
C = CC
MPCC = CC
OPENMP = -openmp #Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
CFLAGS = -O3 -std=c++0x
LIBS =


TARGETS = mpi

all:    $(TARGETS)

mpi: mpi.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o

mpi.o: mpi.cpp common.h MPIVectframe.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt

