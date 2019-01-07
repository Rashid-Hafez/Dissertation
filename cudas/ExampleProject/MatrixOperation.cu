/**-------------------------------------------------------------------------------
Name:

@ Description:
- Program to multiply 2 matrices together.
-------------------------------------------------------------------------------**/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "MatrixOperation.h"
#include <stdio.h>
#define MAX(a,b) (a>b ? a:b)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
int STRIDE;
int PINNED = 0;
int nStreams;
long long GRIDSIZE = (long long)65535L*65535L; //Larger than this does not fit on GPU
int BLOCKSIZE =  32;
int partitions = 2;
static cudaDeviceProp PROPS;
//-------------------------------------------------------------------------------

/** Description:
Row based matrix multiplication with optimised shared memory kernel.

@Parameters:
	- A, B: 1 Dimensional row based arrays
	- C
	- N

**/
__global__ void multiplication(int *A, int* B, int *C, int N){
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; // BlockIndex * BlocksizeY + ThreadY
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   int sum = 0;

   //ZEROED out of bounds indexes
   if (ROW < N && COL < N){

   	for (int i = 0; i < N; ++i)
   	{
   		sum += A[ROW*N+i] * B[i * N + COL]; 
   	}
   	
   	C[ROW*N+COL] = sum;
   }
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void MatrixOperation(int* aC, int* bC, int* cC, long long width1, long long height1, long long width2,
	long long height2, int*a,int*b,int*c, cudaDeviceProp *prop){

	if(width1 != height2){
		printf("Error, width of first mat must = height of second mat\n");
		exit(0);
	}

	/*------------Basic Generic Setup------------------- */
	cudaStream_t stream0;
	gpuErrchk(cudaStreamCreate(&stream0));
	cudaEvent_t start,stop;
	float time;

	dim3 *dimGrid;//Number of blocks
	dim3 *dimBlock;  //number of threads
	/*----------------------------------------------------- */

	if (PINNED) //if page locked memory on matrix
	{	printf("Begin Partitioning\n");
	/*------------------Partition------------------------*/
		unsigned long long SplitA_Row;
		unsigned long long SplitB_Col;
		unsigned long long N;
		unsigned long long MaxData = height1 * width2; //total entries of A and B
		unsigned long long SubMatSize;

		SplitA_Row = ceil (height1/2) ;
		SplitB_Col = ceil (width2/2) ;
		//Make N at Least Half the grid size 
		N = SplitA_Row*SplitB_Col;

		while(N>=GRIDSIZE){
			if (N<GRIDSIZE) //Safety precaution
			{
				printf("N is solved\n");
				break;
			}
			if (N>= GRIDSIZE) //If our matrix is still too big then...
			{
				printf("N BIGGER THAN GRIDSIZE\n");
				SplitA_Row = ceil(SplitA_Row/2);
				SplitB_Col = ceil(SplitB_Col/2);
				N = SplitA_Row * SplitB_Col;
			}
		}

		SetupDim(SplitA_Row, SplitB_Col, &dimGrid, &dimBlock, &prop);
		
		//Timer START LETS GOOO!
		gpuErrchk(cudaEventRecord(start,0));

		gpuErrchk(cudaMalloc((void**)&aC, N));
		gpuErrchk(cudaMalloc((void**)&bC, N));
		gpuErrchk(cudaMalloc((void**)&cC, N));

/*---------------------ASYNC STREAM LOOP------------------------------*/
		for (int i = 0; i < MaxData; i+=N)
		{
			gpuErrchk(cudaMemcpyAsync(aC,a+i,size1,cudaMemcpyHostToDevice,stream0));
			gpuErrchk(cudaMemcpyAsync(bC,b+i,size2,cudaMemcpyHostToDevice,stream0));

			multiplication<<<dimGrid,dimBlock,stream0>>>(aC,bC,cC,height1);

			gpuErrchk(cudaMemcpy(c,cC,size1,cudaMemcpyDeviceToHost,stream0));

		}
	}

	gpuErrchk(cudaStreamSynchronize(stream0)); // Tell CPU to hold his horses and wait

	gpuErrchk(cudaEventRecord(stop,0));
	gpuErrchk(cudaEventSynchronize(stop));
	gpuErrchk(cudaEventElapsedTime(&time, start, stop));

	printf("Time Taken: %3.1f ms/n \n",time );

	gpuErrchk(cudaDestroyStream(stream0));
/*-------------------------------------------------------------------*/

/*--------------------- NO STREAM MULTIPLICATION------------------------------*/
	else{
		long size1 = width1 * height1 * sizeof(int); //matrixA
		long size2 = width2 * height2 * sizeof(int); //matrixB

		printf("Size1 = %d\n",size1);
		printf("Height1 = %d\n",height1);
		
		SetupDim(width1,height2);
		cudaMalloc((void**)&aC, size1);
		cudaMalloc((void**)&bC,size2);
		cudaMalloc((void**)&cC, size2);

		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(c,cC,size1,cudaMemcpyDeviceToHost));
	}
}


/**
**************************************************************************************************
Name: SetupDim

Description:
	Sets up entire grid dimensions. The amount of blocks to use to cover the grid depends on the matrix size.
**************************************************************************************************
**/
void SetupDim (long long width1, long long height2, dim3 * grid, dim3* block, cudaDeviceProp *prop){

	if (prop.major>=2)
	{
		printf("Device compute is 2 or over, utilizing thread count\n");
		int gCol = ceil(width1/BLOCKSIZE);
		int gRow = ceil(height2/BLOCKSIZE);
		printf("Grid is %d by %d \n",gCol,gRow );
		*grid(gCol,gRow);
		*block(BLOCKSIZE,BLOCKSIZE); //(BLOCKSIZE,BLOCKSIZE); //32*32 threads per block. = 1024; Studies suggest this isn't always the most optimal.
	}
	else{
		printf("Device Compute Capacity less than 2, reducing threadcount\n");
		BLOCKSIZE = 16;
		int gCol = ceil(width1/BLOCKSIZE);
		int gRow = ceil(height2/BLOCKSIZE);
		printf("Grid is %d by %d \n", gCol, gRow);
		*grid(gCol,gRow);
		*block(BLOCKSIZE,BLOCKSIZE);
	}

	// if(gridDim.x*blockDim.x < size1){
	// 	STRIDE = 42; // we tell the kernel to use a stride method of multiplication.
	//	}
	//}
		//else if (compute >=3.x){}
		//else if (compute < 2.x){}
}

/**

Description:
Convert normal matrix to ROW MAJOR matrix, if the matrices are bigger than the GPU memory the function will use pinned memory (i.e. hostmalloc). 

a(i,j) can be flatten to 1D array b(k)
mat[0] to mat[m] = the first row, mat[m+1] = the second row. mat[2*m+1] = third row

@Param: 
  - mat : the 2D matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs in the matrix
**/
int * RowMajorMat(int** mat, long long n,long long m){
int * newMat;
unsigned long long ss = n*m;
 if(ss <= GRIDSIZE/10000){
 	newMat = (int*) malloc(ss*sizeof(int));
 } 
 else{
 	printf("Setting up mat for page locked storage\n");
 	if(!SetupMat(&newMat)) return 0;
 }

  for (long i = 0; i<n; i++){
    for (long j =0; j<m; j++){
    long k = i * m + j;
      newMat[k] = mat[i][j];
    }
  }
  return newMat;
}

/**
Convert to Column based 1D array. Depending on the operation either row or column major arrays can lead to more coalesing of threads.
**/
int * ColumnMajorMat(int** mat, long long n, long long m){
int * newMat;
unsigned long long ss = n*m;
 if(ss <= GRIDSIZE){
 	newMat = (int*) malloc(ss*sizeof(int));
 } 
 else{
 	printf("Setting up mat for page locked storage\n");
 	if(!SetupMat(&newMat)) return 0;
 }

  for (long i = 0; i<m; i++){
    for (long j =0; j<n; j++){
    long k = i * n +j;
      newMat[k] = mat[j][i];
    }
  }
  return newMat;
}

//Setup for unified/pinned and others
int SetupMat(int* mat){

	if (!PROPS.canMapHostMemory && !PROPS.managedMemory)
	{
		fprintf(stderr,"Unified AND Pinned memory not supported... exiting\n", );
		return (0);
	}
	int input;
	printf("Enter 1 for pinned, 2 for unified\n");
	scanf("%d",&input);
	if (input ==1) //PINNED MEM
	{ 
		printf("Pinned!\n");
		PINNED =42;
		gpuErrchk(cudaHostAlloc((void**)&mat,ss*sizeof(int),cudaHostAllocDefault)); //Page locked
	}
	if (input == 2) //UNIFIED MEM
	{
		/*
		pinned = true; 
		cudaMallocManaged() */
	}
}

void setProp(int d){
	cudaSetDevice(d);
	cudaGetDeviceProperties(&PROPS,d);
}
cudaDeviceProp getProp(){
	return(PROPS);
}