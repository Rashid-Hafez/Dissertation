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
nvcc -o MatrixExample MatrixExample.cu MatrixOperation.cu
cuda-memcheck MatrixExample

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
int Check(int*a,int*b,int nm, int*c);
/////////////////////////////
struct Matrix
{
  int row;
  int col;
  int p; //partition wanting to send

  int* mat;
}mMatrix;
//////////////////////////////////////////////////////////
void randomInit(int* data, int size)
{
    for (int i = 0; i < size; ++i){
        data[i] = rand()% (1000 + 1 - 1) + 1;
                if(i<5){
                        printf("\n%d",data[i]);
                }
        }
}
//////////////////////////////////////////////////////////
//Basic vector addition. Just here for debugging purposes.
__global__ void vectorAdd(int * aC, int* bC,int* cC){
  
  if(blockIdx.x<N){
    cC[blockIdx.x] = aC[blockIdx.x] + bC[blockIdx.x];
  }
}
///////////////////////////////////////////////////////////////////////////////////////

/**
Description:
Convert normal matrix to ROW MAJOR matrix, if the matrices are bigger than the GPU memory the function will use pinned memory (i.e. hostmalloc). 

a(i,j) can be flatten to 1D array b(k)
mat[0] to mat[m] = the first row, mat[m+1] = the second row. mat[2*m+1] = third row

@Param: 
  - mat : the 2D matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs in the matrix

  MOVE TO SIDE CLASS
**/
int RowMajorMat(int** mat, long long n,long long m, int *&a){

unsigned long long ss = n*m;
int input;
	printf("Enter 1 for pinned, 2 for unified, 3 for normal\n");
	scanf("%d",&input);
	if (input ==1) //PINNED MEM
	{ 
		printf("Pinned!\n");
		SetPinned(42);
		printf("array = %p\n",&a );
		gpuErrchk(cudaHostAlloc((void**)&a,ss*sizeof(int),cudaHostAllocPortable));
    printf("arrayP = %p\n",a );
	}
	if(input==2)
	{
		SetUnified(1);
	}
	if (input==3)
	{
		if(!(a = (int*)malloc(ss*sizeof(int))))return 0;
	}
  for (long i = 0; i<n; i++){
    for (long j =0; j<m; j++){
    long k = i * m + j;
      a[k] = mat[i][j];
    }
  }
  return 1;
}
///////////////////////////////////////////////////////////////////////////////////////
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
  unsigned long long num_rows = N;///ARRAYSIZE(matrixA); //row = sizeof(matrix)/sizeof(matrix[0])
  unsigned long long num_cols = N;//ARRAYSIZE(matrixA[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  unsigned long long num_rows1 = N; //ARRAYSIZE(matrixB); //row = sizeof(matrix)/sizeof(matrix[0])
  unsigned long long num_cols1 = N;//ARRAYSIZE(matrixB[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  
  long long MaxData = (num_rows*num_cols);

  printf("rows = %d \n",num_rows );
  printf("cols = %d\n", num_cols);

  int *a, *b, *c; //host vectors
  gpuErrchk(cudaHostAlloc((void**)&c,(N*N)*sizeof(int),cudaHostAllocPortable));
  printf("a = %p\n",&a );
  if(!(RowMajorMat(matrixA, num_rows,num_cols, a)))fprintf(stderr, "Unable to alocate memory on host\n");
  printf("b = %p\n",&b );
  if(!(RowMajorMat(matrixB, num_rows1,num_cols1,b)))fprintf(stderr, "Unable to alocate memory on host\n");

  free(matrixA); free(matrixB);

  printf("----------------MatrixOperation-------------------------\n");
	MatrixOperation(num_cols,num_rows,num_cols1,num_rows1, &pp);


  /*------------Basic Generic Setup------------------- */
  int* aC;
  int* bC;
  int* cC;
  unsigned long Nn = GetN();
  dim3 GRID = GetGrid();
  dim3 BLOCK = GetBlock();

  printf("N =%d\n",N);
  printf("GRID = %d,%d\n",GRID.x,GRID.y );
  printf("BLOCK = %d,%d\n",BLOCK.x,BLOCK.y );

  cudaStream_t stream0;
  cudaEvent_t start,stop;
  float time;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk( cudaStreamCreate( &stream0 ) );
  //Timer START LETS GOOO!
  gpuErrchk(cudaEventRecord(start,0));
  //malloc
  printf("CudaMalloc\n");
  gpuErrchk(cudaMalloc((void**)&aC, Nn*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&bC, Nn*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&cC, Nn*sizeof(int)));

  printf("a[%d]\n",a[0]);

  printf("----------------------For LOOP--------------------------------\n");
/*---------------------ASYNC STREAM LOOP------------------------------*/
    for (int i = 0; i < MaxData; i+=Nn)
    {
      
      gpuErrchk(cudaMemcpyAsync(aC,a+i,Nn*sizeof(int),cudaMemcpyHostToDevice,stream0));
      gpuErrchk(cudaMemcpyAsync(bC,b+i,Nn*sizeof(int),cudaMemcpyHostToDevice,stream0));
      //                  multiply                  //
      multiplication<<<GRID,BLOCK,0,stream0>>>(aC,bC,cC,N,BLOCK.x);
      gpuErrchk(cudaMemcpyAsync(c+i,cC,Nn*sizeof(int),cudaMemcpyDeviceToHost,stream0)); //i = N;
    }
    gpuErrchk(cudaStreamSynchronize(stream0)); // Tell CPU to hold his horses and wait
    cudaDeviceSynchronize();
    gpuErrchk(cudaEventRecord(stop,0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&time, start, stop));

    printf("Time Taken: %3.1f ms/n \n",time);
    gpuErrchk(cudaStreamDestroy(stream0));


  printf("\n freeing all vectors from memory\n");

  if(!(Check(a,b,N*N,c))){
    gpuErrchk( cudaFreeHost( a ) );
    gpuErrchk( cudaFreeHost( b ) );
    gpuErrchk( cudaFreeHost( c ) );
    gpuErrchk( cudaFree( aC ) );
    gpuErrchk( cudaFree( bC ) );
    gpuErrchk( cudaFree( cC ) );
  }
  
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
      printf("thats wrong bro, index = %d, value should be=%d, but it is = %d\n",i*nm+nm, sum,c[i*nm+nm]);
      return(0);
    }
  }
printf("no error\n");  
return(1);
}
