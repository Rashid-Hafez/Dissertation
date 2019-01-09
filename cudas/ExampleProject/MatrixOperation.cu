/**-------------------------------------------------------------------------------
Name:

@ Description:
- Program to multiply 2 matrices together.
-------------------------------------------------------------------------------**/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "MatrixOperation.h"
#define BLOCKSIZE 32
#define MAX(a,b) (a>b ? a:b)
int STRIDE;
int unified =0;
int PINNED = 0;
int nStreams;
long long GRIDSIZE = (long long)65535L*65535L; //Larger than this does not fit on GPU
int partitions = 2;
int BlockSIZE;
dim3 GRID; dim3 BLOCK;
static cudaDeviceProp PROPS;
static unsigned long N;
//-------------------------------------------------------------------------------

/****************************************** 

@Description: MULT_KERNEL

Row based square matrix multiplication with optimised shared memory kernel.

@Parameters:
	- A, B: 1 Dimensional row based arrays
	- C: result matrix
	- N: Size of row/column

********************************************/
__global__ void multiplication(int *A, int* B, int *C, int N, int BlockSIZE){
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; // BlockIndex * BlocksizeY + ThreadY
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   int sum = 0;
   
   if (ROW < N && COL < N){ //TODO: ZERO OUT OF BOUNDS

   	for (int i = 0; i < N; ++i)
   	{
   		sum += A[ROW*N+i] * B[i * N + COL]; 
   	}
   	C[ROW*N+COL] = sum;
   }
}

void MatrixOperation(long long width1, long long height1, long long width2,
	long long height2, cudaDeviceProp *prop){

	if(width1 != height2){
		printf("Error, width of first mat must = height of second mat\n");
		exit(0);
	}

	if (PINNED) //if page locked memory on matrix
	{	printf("Pinned... Partitioning\n");

	/*------------------Partition------------------------*/
		unsigned long SplitA_Row;
		unsigned long SplitB_Col;
		unsigned long Nn;
		unsigned long long MaxData = height1 * width2; //total entries of A and B
		printf("MaxData = %d\n",MaxData);

		SplitA_Row = ceil (height1/partitions) ;
		SplitB_Col = ceil (width2/partitions) ;
		//Make N at Least Half the grid size 
		Nn = SplitA_Row*SplitB_Col;

		while(Nn>=GRIDSIZE){
			if (Nn<GRIDSIZE) //Safety precaution
			{
				printf("N is solved\n");
				break;
			}
			if (Nn>= GRIDSIZE) //If our matrix is still too big then...
			{
				printf("N BIGGER THAN GRIDSIZE\n");
				SplitA_Row = ceil(SplitA_Row/partitions);
				SplitB_Col = ceil(SplitB_Col/partitions);
				Nn = SplitA_Row * SplitB_Col;
			}
		}

		N = Nn;
		SetupDim(SplitA_Row, SplitB_Col, *prop);
	}
	
/*----------------------------------------------------------------------------*/

else if (unified)
{
	printf("unified\n");
}

/*--------------------- NO STREAM MULTIPLICATION------------------------------*/
	else{
		long size1 = width1 * height1 * sizeof(int); //matrixA
		long size2 = width2 * height2 * sizeof(int); //matrixB

		printf("Size1 = %d\n",size1);
		printf("Height1 = %d\n",height1);

		gpuErrchk( cudaPeekAtLastError() );
	}
}


/**
**************************************************************************************************
Name: SetupDim

Description:
	Sets up entire grid dimensions. The amount of blocks to use to cover the grid depends on the matrix size.
**************************************************************************************************
**/
void SetupDim (long long width1, long long height2, cudaDeviceProp prop){

	if (prop.major>=2)
	{
		int bblock = BLOCKSIZE;
		printf("Device compute is 2 or over, utilizing thread count\n");
		int gCol = ceil((width1+BLOCKSIZE-1)/BLOCKSIZE);
		int gRow = ceil((height2+BLOCKSIZE-1)/BLOCKSIZE);
		// printf("Grid is %d by %d \n",gCol,gRow );
		dim3 grid(gCol,gRow);
		dim3 block(bblock,bblock); //(BLOCKSIZE,BLOCKSIZE) 
		//32*32 threads per block. = 1024 Studies suggest this isn't always the most optimal.
		BlockSIZE = bblock;
		GRID = grid;
		BLOCK = block;
	}
	else{
		printf("Device Compute Capacity less than 2, reducing threadcount\n");
		BlockSIZE = 16;
		int gCol = ceil((width1+BLOCKSIZE-1)/BLOCKSIZE);
		int gRow = ceil((height2+BLOCKSIZE-1)/BLOCKSIZE);
		printf("Grid is %d by %d \n", gCol, gRow);
		dim3 grid(gCol,gRow,1);
		dim3 block(BlockSIZE,BlockSIZE,1); //(BLOCKSIZE,BLOCKSIZE); //32*32 threads per block. = 1024; Studies suggest this isn't always the most optimal.
		
		GRID = grid;
		BLOCK = block;
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
Convert normal matrix to COLUMN MAJOR matrix, if the matrices are bigger than the GPU memory the function will use pinned memory (i.e. cudahostmalloc). 

a(i,j) can be flatten to 1D array b(k)
mat[0] to mat[m] = the first row, mat[m+1] = the second row. mat[2*m+1] = third row

@Param: 
  - mat : the 2D matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs in the matrix
**//*
int ColumnMajorMat(int** mat, long long n, long long m){
int * newMat;
unsigned long long ss = n*m;
 if(ss <= GRIDSIZE){
 	newMat = (int*) malloc(ss*sizeof(int));
 } 
 else{
 	printf("Setting up mat for page locked storage\n");
 	if(!(newMat = SetupMat(ss))) return 0;
 }

  for (long i = 0; i<m; i++){
    for (long j =0; j<n; j++){
    long k = i * n +j;
      newMat[k] = mat[j][i];
    }
  }
  return newMat;
}
*/
void setProp(int d){
	gpuErrchk(cudaSetDevice(d));
	gpuErrchk(cudaGetDeviceProperties(&PROPS,d));
}
cudaDeviceProp getProp(){
	return(PROPS);
}
void SetUnified(int i){
	unified=1;
}
void SetPinned(int i){
	PINNED = 1;
}

int GetPinned(){
	return PINNED;
}
int GetUnified(){
	return unified;
}
unsigned long GetN(){
	return N;
}
dim3 GetGrid(){
	return GRID;
}
dim3 GetBlock(){
	return BLOCK;
}