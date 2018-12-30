/**
Example of main class doing opertations on matrices. This class takes a premade matrces and converts them to 1D array for GPU operations.
**/
#include<time.h>
#include<stdio.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include "MatrixOperation.h"
/////////////// MACROS: //////////////
#define N 30
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ARRAYSIZE(a) (sizeof(a) / sizeof(a[0]))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void randomInit(int* data, int size)
{
    for (int i = 0; i < size; ++i){
        data[i] = rand()% (1000 + 1 - 1) + 1;
                if(i<5){
                        printf("\n%d",data[i]);
                }
        }
}

__global__ void vectorAdd(int * aC, int* bC,int* cC){
  
  if(blockIdx.x<N){
    cC[blockIdx.x] = aC[blockIdx.x] + bC[blockIdx.x];
  }
}

/**
Convert normal matrix to ROW MAJOR matrix. a(i,j) can be flatten to 1D array b(k)
@Param: 
  - mat : matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs
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
  int rowz = 5000;
  int colz = rowz;
  
  int **matrixA = (int**) malloc(rowz * sizeof (int*));
  for (int i = 0; i<rowz; i++){
    matrixA[i] = (int *) malloc(colz * sizeof(int*));
  }
  int **matrixB = (int**) malloc(rowz * sizeof (int*));
  for (int i = 0; i<rowz; i++){
    matrixB[i] = (int *) malloc(colz * sizeof(int*));
  }
 
  for(int i = 0; i<rowz; i++){
    for(int j =0; j<colz; j++){
      matrixA[i][j] = rand()%100)/100.00;
      matrixB[i][j] = rand()%100)/100.00;
    }
  }
   
  // Get row and col size
  int num_rows = ARRAYSIZE(matrixA); //row = sizeof(matrix)/sizeof(matrix[0])
  int num_cols = ARRAYSIZE(matrixA[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  int num_rows1 = ARRAYSIZE(matrixB); //row = sizeof(matrix)/sizeof(matrix[0])
  int num_cols1 = ARRAYSIZE(matrixB[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  int size1 = (num_row*num_col) * sizeof(int)
  RowMajorMat(matrixA, num_rows,num_cols);
  int *a, *b, *c; //host vectors
  a= (int *)malloc(()*sizeof(int));
  b = (int *)malloc(size);
  c=(int *)malloc(size);
  
  int *aC,*bC,*cC;//cuda vectors
  
  struct timespec start,stop;
  printf("\n Code to add vectors A and B");
  
  cudaMalloc((void**)&aC, size);
  cudaMalloc((void**)&bC,size);
  cudaMalloc((void**)&cC, size);

  randomInit(a,N); randomInit(b,N);
  gpuErrchk(cudaMemcpy(aC,a,size,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(bC,b,size,cudaMemcpyHostToDevice));
  clock_gettime(CLOCK_REALTIME,&start);
  //Create kernel of N blocks holding 1 threads
  vectorAdd<<<N,1>>>(aC,bC,cC);//can do <<<N,1>>> for parralel
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(c,cC,size,cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_REALTIME,&stop);
  printf("\n printing 20 results of C");
  for(int i=0;i<20;i++){
    printf("\n%d",c[i]);
  }
  printf("\n freeing all vectors from memory");
  free(a); free(b); free(c);
  cudaFree(aC); cudaFree(bC); cudaFree(cC);//changed to cuda free
  
  return 0;
}
