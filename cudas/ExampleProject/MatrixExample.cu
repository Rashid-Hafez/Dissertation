/**

TODO: 
------------------------
Async Stream
Shared Mem/Tiling
Column based
Matrix Struct
------------------------

Tests to conduct:

- 1 Column based VS row based // Ren Wu(Et.al) study suggests Column based is better than row based (Coaleased)
- 2 SharedMem vs No shared mem
- 3 Stream vs non stream 
- 4 Stream Count    // Studies suggest 2 streams most optimal
- 5 Unified vs Pinned //I think Unified will be faster

- 6 Thread Count/Blocksize/BlockCount. This needs to be adapted for every test.
- 7 B =Column based  A =row based. vicea versa


Notes:
-------------------------------------------
Maximum number of threads per block: 1,024
Maximum sizes of each dimension of a block: 1,024 × 1,024 × 64,
Because 1,024 is the upper limit for the number of threads in a block, the largest 2D block is: 32 × 32 == 1,024

Maximum sizes of each dimension of a grid: 65,535 × 65,535 × 65,535 
65,535 is the upper limit for the builtin variables suchas gridDim.x, gridDim.y, gridDim.z


    blockDim.x,y,z gives the number of threads in a block, in the particular direction
    gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
    blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case)


Calculate peak bandwidth:
Memory Clock * 


CANT READ AND WRITE ON SAME LINE. MUST USE SYNCTHREADS... 
so array[idx] = array [idx+1] WRONG!!!

   //must do this: temp = array[idx+1];
   //syncthreads()
   //array[idx] = temp;
   //syncthreads();
  // COALESE thrreads not stride, if threads acess farr apart. 
  //put each far apart in 1 array and process it so they all access close, then you can put them babck in the original array.

  @ References:
  - Cuda By Example
  - HP
  - Stencil Kernel
  - 

**/

/**
Example of main class doing opertations on matrices. This class takes a premade matrces and converts them to 1D array for GPU operations.
**/
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cuda.h>
#include <cuda_runtime.h>
#include "MatrixOperation.h"
// #include "Example.h"

/////////////// MACROS and GLOBALS: //////////////
#define N 4
#define ARRAYSIZE(a) (sizeof(a) / sizeof(a[0])) //will only work with stationary pointer
/////////////////////////////
struct Matrix
{
  int row;
  int col;
  int p; //partition wanting to send

  int* mat;
}mMatrix;

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


int main(int argc, char** argv){

  //Setup Check//
  int Dev = 0;
  cudaDeviceProp pp;
  setProp(Dev);
  pp = getProp();
//-----------------------------------------------------------

  srand(356);

  printf("Initialised\n");
  printf("Creating Template Matrix\n");
  int rowz = N;
  int colz = rowz;
  printf("colz = %d\n",colz );

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
      matrixA[i][j] = rand()%100;
      matrixB[i][j] = rand()%100;
    }
  }

  printf("Matrix before RowMajor: \n");
  printf("MatrixA row1:%d,%d,%d \n",matrixA[0][0],matrixA[0][1],matrixA[0][2]);
  printf("MatrixA row2:%d,%d,%d \n",matrixA[1][0],matrixA[1][1],matrixA[1][2]);
  printf("MatrixB row1:%d,%d,%d \n",matrixB[0][0],matrixB[0][1],matrixB[0][2]);
  printf("MatrixB row2:%d,%d,%d \n",matrixB[1][0],matrixB[1][1],matrixB[1][2]);

  // Get row and col size
  int num_rows = N;///ARRAYSIZE(matrixA); //row = sizeof(matrix)/sizeof(matrix[0])
  int num_cols = N;//ARRAYSIZE(matrixA[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  int num_rows1 = N; //ARRAYSIZE(matrixB); //row = sizeof(matrix)/sizeof(matrix[0])
  int num_cols1 = N;//ARRAYSIZE(matrixB[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  
  int size1 = (num_rows*num_cols) * sizeof(int); // for malloc and memcpy
  //int size2 = (num_rows*num_cols) * sizeof(int); // for malloc and memcpy

  printf("size1 = %d\n", size1);
  printf("rows = %d \n",num_rows );
  printf("cols = %d\n", num_cols);

  int *a, *b, *c; //host vectors
  if(!(a= RowMajorMat(matrixA, num_rows,num_cols)))fprintf(stderr, "Unable to alocate memory on host\n");
  if(!(b = RowMajorMat(matrixB, num_rows1,num_cols1)))fprintf(stderr, "Unable to alocate memory on host\n");

  free(matrixA); free(matrixB);
  
  int *aC,*bC,*cC;//cuda vectors

	MatrixOperation(aC, bC, cC,num_cols,num_rows,num_cols1,num_rows1, a, b, c, &pp);

  printf("\n freeing all vectors from memory");
  printf("Verifying...\n", );

  Check(c);

  free(a); free(b); free(c);
  cudaFree(aC); cudaFree(bC); cudaFree(cC);//changed to cuda free
  
  return 0;
}

/**
  Verify if multiplication output is correct
**/
int Check(int* a1, int* b1, int nm, int* c){

  int sum;
  for (int i = 0; i < nm; ++i)
  {
    for (int j = 0; j < nm; ++j)
    {
      sum+= a1[i*nm+j] * b1[j*nm+nm]; 
    }

    if(c[i*nm+nm]!=sum){
      printf("thats wrong bro, index = %d\n",i*nm+j);
    }
  }

}
