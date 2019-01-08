NVCC=nvcc
CUDAFLAGS= -arch=sm_30
OPT= -g -G
RM=/bin/rm -f

all: MatrixExample

main: MatrixExample.o MatrixOperation.o

        ${NVCC} ${OPT} -o main MatrixExample.o MatrixOperation.o

MatrixOperation.o: MatrixOperation.h MatrixOperation.cpp

        ${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c Generate.cpp

MatrixExample.o: MatrixExample.cu

        $(NVCC) ${OPT} $(CUDAFLAGS)        -std=c++11 -c MatrixExample.cu

MatrixExample: MatrixExample.o MatrixOperation.o

        ${NVCC} ${CUDAFLAGS} -o MatrixExample MatrixExample.o MatrixOperation.o

clean:

        ${RM} *.o MatrixExample