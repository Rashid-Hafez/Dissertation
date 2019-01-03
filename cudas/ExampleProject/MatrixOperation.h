#ifndef MATRIXOPERATION_H_   /* Include guard */
#define MATRIXOPERATION_H_
#include <stdlib.h>

void MatrixOperation(int* aC, int* bC, int* cC, long width1, long height1, long width2, long height2, int*a,int*b,int*c, cudaDeviceProp* prop);  

__global__ void multiplication(int *A, int* B, int *C, int N);

void SetupDim (long width1, long height1, long width2, long height2, dim3*grid,dim3*block, cudaDeviceProp* prop, int init);

int * RowMajorMat(int** mat, long n, long m);
#endif // MATRIXOPERATION_H_
