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

__global__ void multiplication(int *A, int* B, int* C, int N, int BlockSIZE);

void SetupDim (long long width1, long long height2,cudaDeviceProp prop);

int ColumnMajorMat(int** mat, long long n, long long m);
void setProp(int d);
cudaDeviceProp getProp();
int* SetupMat(long long size);
void SetPinned(int i);
void SetUnified(int i);
int GetUnified();
int SetUnified();
unsigned long GetN();
dim3 GetGrid();
dim3 GetBlock();
#endif // MATRIXOPERATION_H_
