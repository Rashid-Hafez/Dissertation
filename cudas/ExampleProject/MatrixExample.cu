/**
Example of main class doing opertations on matrices. This class takes a premade matrces and converts them to 1D array for GPU operations.
**/
#include<time.h>
#include<stdio.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include "MatrixOperation.h"
// #include "Example.h"

/////////////// MACROS: //////////////
#define N 4
#define ARRAYSIZE(a) (sizeof(a) / sizeof(a[0]))
/////////////////////////////

void randomInit(int* data, int size)
{
    for (int i = 0; i < size; ++i){
        data[i] = rand()% (1000 + 1 - 1) + 1;
                if(i<5){
                        printf("\n%d",data[i]);
                }
        }
}

//Basic vector addition. Just here for debugging purposes.
__global__ void vectorAdd(int * aC, int* bC,int* cC){
  
  if(blockIdx.x<N){
    cC[blockIdx.x] = aC[blockIdx.x] + bC[blockIdx.x];
  }
}

/**
Convert normal matrix to ROW MAJOR matrix. a(i,j) can be flatten to 1D array b(k)
mat[0] to mat[m] = the first row, mat[m+1] = the second row. mat[2*m+1] = third row
@Param: 
  - mat : the 2D matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs in the matrix
**/
int * RowMajorMat(int** mat, int n, int m){
  int * newMat = (int *) malloc ((n*m)*sizeof(int));
  for (int i = 0; i<n; i++){
    for (int j =0; j<m; j++){
    int k = i * m + j;
      newMat[k] = mat[i][j]
    }
  }
  return newMat;
}

int main(){
  
  printf("Initialised\n");
  printf("Creating Template Matrix\n");
  int rowz = N;
  int colz = rowz;
  
  int **matrixA = (int**) malloc(rowz * sizeof (int*));
  for (int i = 0; i<rowz; i++){
    matrixA[i] = (int *) malloc(colz * sizeof(int));
  }
  int **matrixB = (int**) malloc(rowz * sizeof (int*));
  for (int i = 0; i<rowz; i++){
    matrixB[i] = (int *) malloc(colz * sizeof(int));
  }
 
  for(int i = 0; i<rowz; i++){
    for(int j =0; j<colz; j++){
      matrixA[i][j] = rand()%100/100.00;
      matrixB[i][j] = rand()%100/100.00;
    }
  }

  printf("Matrix before RowMajor: \n");
  printf("MatrixA row1:%d,%d,%d \n",matrixA[0][0],matrixA[0][1],matrixA[0][2]);
  printf("MatrixA row2:%d,%d,%d \n",matrixA[1][0],matrixA[1][1],matrixA[1][2]);
  printf("MatrixB row1:%d,%d,%d \n",matrixB[0][0],matrixB[0][1],matrixB[0][2]);
  printf("MatrixB row2:%d,%d,%d \n",matrixB[1][0],matrixB[1][1],matrixB[1][2]);

  // Get row and col size
  int num_rows = ARRAYSIZE(matrixA); //row = sizeof(matrix)/sizeof(matrix[0])
  int num_cols = ARRAYSIZE(matrixA[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  int num_rows1 = ARRAYSIZE(matrixB); //row = sizeof(matrix)/sizeof(matrix[0])
  int num_cols1 = ARRAYSIZE(matrixB[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  
  int size1 = (num_rows*num_cols) * sizeof(int); // for malloc and memcpy
  int size2 = (num_rows*num_cols) * sizeof(int); // for malloc and memcpy

  printf("size1 = %d\n", size1);
  printf("size2 = %d\n", size2);

  int *a, *b, *c; //host vectors
  a= RowMajorMat(matrixA, num_rows,num_cols);
  b = RowMajorMat(matrixB, num_rows1,num_cols1);
  c=(int *)malloc(size1);

  printf("Size of size1 = %d\n",size1);
  printf("Matrix AFTER RowMajor: \n");
  printf("MatrixA row1:%d,%d,%d \n",a[0],a[1],a[2]);
  printf("MatrixA row2:%d,%d,%d \n",a[N+1],a[N+2],a[N+3]);
  printf("MatrixB row1:%d,%d,%d \n",b[0],b[1],b[2]);
  printf("MatrixB row2:%d,%d,%d \n",b[N+1],b[N+2],b[N+3]);
  
  int *aC,*bC,*cC;//cuda vectors

	MultiplyMatrix(aC, bC, cC,num_cols,num_rows,num_cols1,num_rows1, a, b, c);
  
  printf("\n Result:");
  
  for(int i=0;i<20;i++){
  printf("MatrixC row1:%d,%d,%d \n",c[0],c[1],c[2]);
  printf("MatrixC row2:%d,%d,%d \n",c[N+1],c[N+2],c[N+3]);
  }

  printf("\n freeing all vectors from memory");
  free(a); free(b); free(c);
  cudaFree(aC); cudaFree(bC); cudaFree(cC);//changed to cuda free
  
  return 0;
}
/**

Notes:

Maximum number of threads per block: 1,024
Maximum sizes of each dimension of a block: 1,024 × 1,024 × 64,
Because 1,024 is the upper limit for the number of threads in a block, the largest 2D block is: 32 × 32 == 1,024

Maximum sizes of each dimension of a grid: 65,535 × 65,535 × 65,535 
65,535 is the upper limit for the builtin variables suchas gridDim.x, gridDim.y, gridDim.z


    blockDim.x,y,z gives the number of threads in a block, in the particular direction
    gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
    blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case)

Striding is: 


**/
