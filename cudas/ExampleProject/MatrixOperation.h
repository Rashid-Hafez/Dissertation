#ifndef MATRIXOPERATION_H_   /* Include guard */
#define MATRIXOPERATION_H_
#include <stdlib.h>
#include <stdio.h>      /* printf, NULL */
#define BLOCKSIZE 32
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	printf("assert\n");
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void MatrixOperation(long long width1, long long height1, long long width2, long long height2, cudaDeviceProp* prop);  

__global__ void vectorAdd(int * aC, int* bC,int* cC,int n);
__global__ void BIG_DOT(long long *A, long long* B, long long *C, unsigned long aCol, unsigned long bRow);
__global__ void multiplication(long long *A, long long * B, long long *C, unsigned long long N1);
__global__ void MatrixMulCUDA(long long *C,long long *A,long long *B,unsigned long long wA,unsigned long long wB);
__global__ void multiplicationR(long long *A, long long* B, long long *C, unsigned long long aCol,unsigned long long bRow);

void SetupDim (long long width1, long long height2,cudaDeviceProp prop);
int ColMajorMat(long long** mat, unsigned long long n,unsigned long long m, long long *&b);
void setProp(int d);
cudaDeviceProp getProp();
void SetPinned(int i);
void SetUnified(int i);
int GetUnified();
int SetUnified();
unsigned long long GetN();
unsigned long long getRow();
unsigned long long getCol();
dim3 GetGrid();
dim3 GetBlock();

#endif // MATRIXOPERATION_H_
