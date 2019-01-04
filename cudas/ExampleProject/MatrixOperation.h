#ifndef MATRIXOPERATION_H_   /* Include guard */
#define MATRIXOPERATION_H_
#include <stdlib.h>

void MatrixOperation(int* aC, int* bC, int* cC, long long width1, long long height1, long long width2, long long height2, int*a,int*b,int*c, cudaDeviceProp* prop);  

__global__ void multiplication(int *A, int* B, int *C, int N);

void SetupDim (long long width1, long long height2, dim3*grid,dim3*block, cudaDeviceProp* prop);

int * RowMajorMat(int** mat, long long n, long long m);
int * ColumnMajorMat(int** mat, long long n, long long m);
void setProp(int d);
cudaDeviceProp getProp();
#endif // MATRIXOPERATION_H_
