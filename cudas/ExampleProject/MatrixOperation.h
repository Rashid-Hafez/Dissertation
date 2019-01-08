#ifndef MATRIXOPERATION_H_   /* Include guard */
#define MATRIXOPERATION_H_
#include <stdlib.h>
#define BLOCKSIZE 32

void MatrixOperation(int* aC, int* bC, int*cC, long long width1, long long height1, long long width2, long long height2, int a,int b,int c, cudaDeviceProp* prop);  

__global__ void multiplication(int *A, int* B, int* C, int N, int BlockSIZE);

void SetupDim (long long width1, long long height2,cudaDeviceProp prop);

int * RowMajorMat(int** mat, long long n, long long m);
int * ColumnMajorMat(int** mat, long long n, long long m);
void setProp(int d);
cudaDeviceProp getProp();
int* SetupMat(long long size);

#endif // MATRIXOPERATION_H_
