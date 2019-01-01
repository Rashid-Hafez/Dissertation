NVCC=nvcc
CUDAFLAGS= -arch=sm_30
OPT= -g -G
RM=/bin/rm -f

all: IC

main: IC.o Generate.o

        ${NVCC} ${OPT} -o main IC.o Generate.o

Generate.o: Header.cuh Generate.cpp

        ${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c Generate.cpp

IC.o: Header.cuh IC.cu

        $(NVCC) ${OPT} $(CUDAFLAGS)        -std=c++11 -c IC.cu

IC: IC.o Generate.o

        ${NVCC} ${CUDAFLAGS} -o IC IC.o Generate.o

clean:

        ${RM} *.o IC