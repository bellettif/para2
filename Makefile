#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CC = CC
MPCC = CC
OPENMP = -openmp #Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
CFLAGS = -O3
LIBS =


TARGETS = serial pthreads openmp mpi autograder

all:	$(TARGETS)

serial: serial.o common.o quadtree.o
	$(CC) -o $@ $(LIBS) serial.o common.o quadtree.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o
pthreads: pthreads.o common.o
	$(CC) -o $@ $(LIBS) -lpthread pthreads.o common.o
openmp: openmp.o common.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o common.o
mpi: mpi.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o

autograder.o: autograder.cpp common.h
	$(CC) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h quadtree.h
	$(CC) -c $(CFLAGS) serial.cpp
pthreads.o: pthreads.cpp common.h
	$(CC) -c $(CFLAGS) pthreads.cpp
mpi.o: mpi.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp
quadtree.o: quadtree.cpp quadtree.h common.h
	$(CC) -c $(CFLAGS) quadtree.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
