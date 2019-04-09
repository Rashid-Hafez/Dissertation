/**

Strassen's Matrix Multiplication TO split

ASK:

  usually matrices are given in row major ordering. Would it be practicle to convert row to column or just assume B is in column
  order already?

  should i do program flexibility... (for example: if !thisProperty then threads=3498 )

  should i just start with very large transpose first?
  
  Should i even transpose B?
  2 KERNELS?.. transpose B to column because each column start is miles apart
  Has it become a Vector Calculation now... because the 2d aspect of the 1D array doesnt fit. only half the row fits.
  
  Should I keep it as a row & column problem. or just a vector row problem... do you think it can be solved in a 2d fasion?


TODO: 
------------------------
Async Stream
Shared Mem/Tiling
Column based
Matrix Struct
MallocPitched
2d access in kernel
------------------------

Tests to conduct:

- 1 Column based VS row based // Ren Wu(Et.al) study suggests Column based is better than row based (Coaleased)
- 2 SharedMem vs No shared mem
- 3 Stream vs non stream 
- 4 Stream Count    // Studies suggest 2 streams most optimal
- 5 Unified vs Pinned //I think Unified will be faster
- 6 Thread Count/Blocksize/BlockCount. This needs to be adapted for every test.

- 7: make B column based at the start vs make it row based, then add the column indexes in order to a seperate array and send to GPU.


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

  - export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/Developer/NVIDIA/CUDA-10.0/lib
  - export PATH=/Developer/NVIDIA/CUDA-10.0/bin${PATH:+:${PATH}}
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
int Check(long long*a,long long*b,unsigned long long nm, long long*c);
int CheckR(long long* a1, long long* b1, unsigned long long nm, long long* c);
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
int RowMajorMat(long long** mat, unsigned long long n,unsigned long long m, long long *&a){

unsigned long long ss = n*m;
int input;
	printf("Enter 1 for pinned, 2 for unified, 3 for normal\n");
	scanf("%d",&input);
	if (input ==1) //PINNED MEM
	{ 
		printf("Pinned!\n");
		SetPinned(42);
		printf("array = %p\n",&a );
    printf("arrayP = %p\n",a );
	}
	if(input==2)
	{
		SetUnified(1);
	}
	if (input==3)
	{
		if(!(a = (long long*)malloc(ss*sizeof(unsigned long long))))return 0;
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

  long long **matrixA = (long long**) malloc(N * sizeof (long long*));
  for (int i = 0; i<N; i++){
    matrixA[i] = (long long *) malloc(N * sizeof(unsigned long long));
  }
  long long **matrixB = (long long**) malloc(N * sizeof (long long*));
  for (int i = 0; i<N; i++){
    matrixB[i] = (long long *) malloc(N * sizeof(unsigned long long));
  }

  for(long i = 0; i<N; i++){
    for(long j =0; j<N; j++){
      matrixA[i][j] = rand()%100;
      matrixB[i][j] = rand()%100;
    }
  }

  // printf("Matrix before RowMajor: \n");
  // printf("MatrixA row1:%d,%d,%d \n",matrixA[0][0],matrixA[0][1],matrixA[0][2]);
  // printf("MatrixA row2:%d,%d,%d \n",matrixA[1][0],matrixA[1][1],matrixA[1][2]);
  // printf("MatrixB row1:%d,%d,%d \n",matrixB[0][0],matrixB[0][1],matrixB[0][2]);
  // printf("MatrixB row2:%d,%d,%d \n",matrixB[1][0],matrixB[1][1],matrixB[1][2]);

  // Get row and col size
  unsigned long long num_rows = N;///ARRAYSIZE(matrixA); //row = sizeof(matrix)/sizeof(matrix[0])
  unsigned long long num_cols = N;//ARRAYSIZE(matrixA[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  unsigned long long num_rows1 = N; //ARRAYSIZE(matrixB); //row = sizeof(matrix)/sizeof(matrix[0])
  unsigned long long num_cols1 = N;//ARRAYSIZE(matrixB[0]);  //col = sizeof(matrix[0])/sizeof(matrix[0][0])
  
  unsigned long long MaxData = (num_rows*num_cols);

  printf("rows = %llu \n",num_rows );
  printf("cols = %llu\n", num_cols);

  //host vectors
  long long * a; 
  long long * b;
  long long * c;

  gpuErrchk(cudaHostAlloc((void**)&c,((MaxData)*sizeof(unsigned long long)),cudaHostAllocPortable));
  gpuErrchk(cudaHostAlloc((void**)&a,((MaxData)*sizeof(unsigned long long)),cudaHostAllocPortable));
  gpuErrchk(cudaHostAlloc((void**)&b,((MaxData)*sizeof(unsigned long long)),cudaHostAllocPortable));

  printf("a = %p\n",&a );
  if(!(RowMajorMat(matrixA, num_rows,num_cols, a)))fprintf(stderr, "Unable to alocate memory on host\n");
  printf("b = %p\n",&b );
  if(!(RowMajorMat(matrixB,num_rows1,num_cols1, b)))fprintf(stderr, "Unable to alocate memory on host\n");

  printf("----------------MatrixOperation-------------------------\n");
	MatrixOperation(num_cols,num_rows,num_cols1,num_rows1, &pp);
  /*------------Basic Generic Setup------------------- */
  long long * aC;
  long long * bC;
  long long * cC;

  unsigned int Nn = GetN(); // Partition

  dim3 GRID = GetGrid();
  dim3 BLOCK = GetBlock();

  printf("N =%llu\n",Nn);
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
  gpuErrchk(cudaMalloc((void**)&aC, (MaxData*sizeof( long long))));
  gpuErrchk(cudaMalloc((void**)&bC, (MaxData*sizeof( long long))));
  gpuErrchk(cudaMalloc((void**)&cC, (MaxData*sizeof( long long))));

  printf("a[%lld]\n",a[0]);

   cudaDeviceSynchronize();
   //gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, Nn));
   unsigned long long r = getRow(); //wA //wB
   unsigned long long col = getCol();

   printf("r = %llu, col = %llu \n",r,col );

  /*
    If rows in A bigger than. num_colsA=rowSize. r=halfRowsize. We may have to send N  A and B and mult them
    then add it to C[0]. then send it back. and do it again. then when C[0] is complete. send the next row of and and B
    if(A_Row % N == 0)
  */

   printf("----------------------For LOOP--------------------------------\n");

   gpuErrchk(cudaMemcpyAsync(matrixA,(a),(MaxData*sizeof(long long)),cudaMemcpyHostToDevice,stream0));
   gpuErrchk(cudaMemcpyAsync(bC,(b),(MaxData*sizeof(long long)),cudaMemcpyHostToDevice,stream0));

   multiplicationR<<<GRID,BLOCK,0,stream0>>>(aC,bC,cC,r,col);
   
   gpuErrchk(cudaMemcpyAsync(c,cC,(MaxData*sizeof(long long)),cudaMemcpyDeviceToHost,stream0)); //i = N;



   /*---------------------ASYNC STREAM LOOP------------------------------*/
    // for (int i = 0; i < MaxData; i+=Nn)
    // {
     
    //   gpuErrchk(cudaMemcpyAsync(aC,(a+i),(Nn*sizeof(int)),cudaMemcpyHostToDevice,stream0));
    //   gpuErrchk(cudaMemcpyAsync(bC,(b+i),(Nn*sizeof(int)),cudaMemcpyHostToDevice,stream0));
    //  // printf("i = %d, a = %d, &a= %p, b= %d, &b= %p, c= %d, &c= %p\n",i, a[i], (&a)+i, b[i], (&b)+i, c[i], (&c)+i );
    //   multiplication<<<GRID,BLOCK,0,stream0>>>(aC,bC,cC,r,BLOCK.x);
    //   gpuErrchk(cudaMemcpyAsync((c+i),cC,(Nn*sizeof(int)),cudaMemcpyDeviceToHost,stream0)); //i = N;
    // }

    gpuErrchk(cudaStreamSynchronize(stream0)); // Tell CPU to hold his horses and wait
    cudaDeviceSynchronize();
    gpuErrchk(cudaEventRecord(stop,0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&time, start, stop));

    printf("Time Taken: %3.1f ms/n \n",time);
    gpuErrchk(cudaStreamDestroy(stream0));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

  printf("\n freeing all vectors from memory\n");

  if(!(Check(a,b,N,c))){

  }
  else{

  }
    gpuErrchk( cudaFreeHost( a ) );
    gpuErrchk( cudaFreeHost( b ) );
    gpuErrchk( cudaFreeHost( c ) );
    gpuErrchk( cudaFree( aC ) );
    gpuErrchk( cudaFree( bC ) );
    gpuErrchk( cudaFree( cC ) );
  return 0;
}
/**
  Verify if multiplication output is correct
**/
int Check(long long* a1, long long* b1, unsigned long long nm, long long* c){

  for (unsigned long long i = 0; i < nm; ++i) 
    {
        for (unsigned long long j = 0; j < nm; ++j) 
        {
            unsigned long long sum = 0.0;
            for (unsigned long long h = 0; h < nm; ++h) 
            {
                sum += a1[i * nm + h] * b1[h * nm + j];
            }
            if(c[i * nm + j] != sum)
            {
              printf("thats wrong bro, index = %d, value should be=%d, but it is = %d\n",i*nm+j,sum,c[i*nm+j]);
            }
        }
}
printf("no error\n");  
return(1);
}

/**
  Verify if multiplication output is correct for when B is COL BASED
**/
int CheckR(long long* a1, long long* b1, unsigned long long nm, long long* c){

  for (unsigned long long i = 0; i < nm; ++i) 
    {
        for (unsigned long long j = 0; j < nm; ++j) 
        {
            unsigned long long sum = 0.0;
            for (unsigned long long h = 0; h < nm; ++h) 
            {
                sum += a1[i * nm + h] * b1[i * nm + h];
            }
            if(c[i * nm + j] != sum)
            {
              printf("thats wrong bro, index = %d, value should be=%d, but it is = %d\n",i*nm+j,sum,c[i*nm+j]);
            }
        }
}
printf("no error\n");  
return(1);
}

