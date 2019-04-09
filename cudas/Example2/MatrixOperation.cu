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
int partitions = 2;//2 = 4 partitions
int BlockSIZE;
dim3 GRID; dim3 BLOCK;
static cudaDeviceProp PROPS;
static unsigned long long N;
static unsigned long long aRow;
static unsigned long long bCol;
//-------------------------------------------------------------------------------

//Basic vector addition. Just here for debugging purposes.
__global__ void vectorAdd(int * aC, int* bC,int* cC,int n){
  
  if(blockIdx.x<n){
    cC[blockIdx.x] = aC[blockIdx.x] + bC[blockIdx.x];
  }
}


/****************************************** 

@Description: BASIC SQUARED MULT_KERNEL

@Parameters:
	- A: 1 Dimensional ROW based arrays
	- B: 1 Dimensional COL based arrays
	- C: result matrix
	- N: Size of row/column

********************************************/
__global__ void multiplicationR(long long *A, long long* B, long long *C, unsigned long long aCol,unsigned long long bRow){
   
   int ROW = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned long long sum = 0;
   
   while (ROW < aCol ){ //

   	for (unsigned long long i = 0; i < aCol; i++)
   	{
   		sum += A[ROW*aCol+i] * B[ROW * bRow + i];
   	}
   	C[ROW*bRow] = sum;
   	ROW += blockDim.y * gridDim.y;
   }
}


/****************************************** 
@Description: BIG_DOT

Row based square matrix multiplication with optimised shared memory kernel.

Each block has own private copy of shared memory. 

REDUCTION STEP:

We call the general process of taking an input array and performing some computations that produce a smaller array of results a reduction.

The naïve way to accomplish this reduction would be having one thread iterate over the shared memory 
and calculate a running sum. This will take us time proportional to the length of the array.
O(Log N) for reduction step.

Reduction performed on __shared__ CACHE[]

@Parameters:
	- A: 1 Dimensional row based array
	- B: 1 Dimensional column based array
	- C: result matrix
	- N: Size of row/column
********************************************/
__global__ void BIG_DOT(long long *A, long long* B, long long *C, unsigned long aCol, unsigned long bRow) //**a,**b,**c
	{

	__shared__ float cache[BLOCKSIZE][BLOCKSIZE]; //Store result of each thread in each block. There is 1 instance of this in each block that all threads share

  unsigned int ROW = blockIdx.y*blockDim.y+threadIdx.y; // Index of a thread's Y position in the grid of blocks
  unsigned int COL = blockIdx.x*blockDim.x+threadIdx.x; // Index of same thread's X position in the grid of blocks

  unsigned int threadINDEX = threadIdx.y*blockDim.y+threadIdx.x; // index of thread in BLOCK. Stored in row order array. so we need to access like a row order.

  int xx = threadIdx.x;
  int yy = threadIdx.y;

   int sum = 0; // move into while loop?

   	while(ROW<aCol && COL<bRow) //disa makea no sense
   	{
   		for (int i = 0; i < bRow; ++i)
   		{
   			sum += A[ROW*aCol+i] * B[i * bRow + COL];  //COL IS LIMITED TO COL GRIDSIZE NOT N SIZE... make B colomn major. 0*4 + i
   		}

   		ROW += blockDim.y * gridDim.y; //Increment all threads at their position by the blocksize.
   		COL += blockDim.x * gridDim.x; //Increment all threads at thier position by the blocksize
   	}
   	//Store result of THIS thread in Block memory
   	cache[xx][yy]= sum; //all threads in this block can access
   	__syncthreads();

   	//The general idea is that each thread will add two of the values in cache[] and store the result back to cache[]. 
   	//Since each thread combines two entries into one, we complete this step with half as many entries as we started with.
//*basically* all the threads greater than 0 will compute a reduction and store it backwards until it reaches 0.
   	int i = (blockDim.x*blockDim.y)/2;
	while (i != 0) 
	{
		if (threadINDEX < i){ // In order to prevent the higher threads that completed from looping again
			cache[xx][yy] += cache[xx + i][yy + i];
		}
		__syncthreads(); //this allows the first thread to read the new values in the cache, which have been computed by other threads.
			i /= 2; // now we want to compute with the threads before this current thread
	}

	if (threadINDEX == 0) //the first thread in each block. Obviously there are multiple threads with an index of 0 (in refrence to its position in a block)
	{
		//this part is hard... how would we store it?....
   		C[blockIdx.x] = cache[0][0];
   	}
  }
/****************************************** 

@Description: BASIC SQUARED MULT_KERNEL

@Parameters:
	- A, B: 1 Dimensional row based arrays
	- C: result matrix
	- N: Size of row/column

********************************************/
__global__ void multiplication(long long *A, long long* B, long long *C, unsigned long long N1){ //**a,**b,**c
   
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; // 
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   
   unsigned long long sum = 0;
   
   if (ROW < N1 && COL < N1){ //

   	for (unsigned long long i = 0; i < N1; i++)
   	{
   		sum += A[ROW*N1+i] * B[i * N1 + COL];  //offset by N to get column
   	}
   	C[ROW*N1+COL] = sum;
   }
}

/****************************************** 

@Description: MatrixMultCUDA

@Parameters:
	- A, B: 1 Dimensional row based arrays
	- C: result matrix
	- N: Size of row/column

********************************************/
__global__ void MatrixMulCUDA(long long *C, long long *A, long long *B, unsigned long long wA, unsigned long long wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
   unsigned long long aBegin = wA * BLOCKSIZE * by; // 4 * 32 * 0

    // Index of the last sub-matrix of A processed by the block
    unsigned long long aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCKSIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCKSIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    unsigned long long bStep  = BLOCKSIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix

    for (unsigned long long a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCKSIZE][BLOCKSIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll //unrolls a loop, which has been predetermined in size...

        for (int k = 0; k < BLOCKSIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    unsigned long long c = wB * BLOCKSIZE * by + BLOCKSIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/****************************************** 

@Description: CONSTRUCTOR

Basic Check and Setup:

@Parameters:
	- width1, width2: how many columns are in the matrix 'A' and 'B'
	- C: result matrix
	- N: Size of row/column

********************************************/
void MatrixOperation(long long width1, long long height1, long long width2,
	long long height2, cudaDeviceProp *prop){

	if(width1 != height2){
		printf("Error, width of first mat must = height of second mat\n");
		exit(0);
	}

	if (PINNED) //if page locked memory on matrix
	{	printf("Pinned... Partitioning\n");

	/*------------------Partition------------------------*/
		unsigned long SplitA_Col;
		unsigned long SplitB_Row;
		unsigned long Nn;
		unsigned long long MaxData = width1 * height2; //total entries of A and B
		printf("MaxData = %d\n",MaxData);

		SplitA_Col = ceil (width1/partitions) ;
		SplitB_Row = ceil (height2/partitions) ;
		printf("width = %d\n", SplitA_Col);
		//Make N at Least Half the grid size 
		Nn = MaxData;

		if(Nn>GRIDSIZE){
			while(Nn>=GRIDSIZE){
				if (Nn<GRIDSIZE) //Safety precaution
				{
					printf("N is solved\n");
					N = Nn;
					SetupDim(SplitA_Col,SplitB_Row, *prop);
					break;
				}
				if (Nn>= GRIDSIZE) //If our matrix is still too big then...
				{
					printf("N BIGGER THAN GRIDSIZE\n");
					SplitA_Col = ceil(SplitA_Col/partitions);
					SplitB_Row = ceil(SplitB_Row/partitions);
					Nn = SplitA_Col * SplitB_Row;
				}
			}
		
			long overflow = MaxData%Nn;
			if (!overflow)
			{
				/* Partition fits perfectly */
			}
			else{

			}
		}
		
		//N is smaller than max size
		else{
			N = Nn;
			aRow = width1;
			bCol = height2;
			SetupDim(SplitA_Col, SplitB_Row, *prop);
		}
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
		dim3 grid(gCol,gRow);
		dim3 block(BlockSIZE,BlockSIZE); //(BLOCKSIZE,BLOCKSIZE); //32*32 threads per block. = 1024; Studies suggest this isn't always the most optimal.
		
		GRID = grid;
		BLOCK = block;
	}
}

/******************************************************************
Description:
Convert normal matrix to COLUMN MAJOR matrix, if the matrices are bigger than the GPU memory the function will use 
pinned memory (i.e. cudahostmalloc). 

a(i,j) can be flatten to 1D array b(k)
mat[0] to mat[m] = the first row, mat[m+1] = the second row. mat[2*m+1] = third row

@Param: 
  - mat : the 2D matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs in the matrix
******************************************************************/
int ColMajorMat(long long** mat, unsigned long long n,unsigned long long m, long long *&b){
int * newMat;
unsigned long long ss = n*m;

int input;
printf("Enter 1 for pinned, 2 for unified, 3 for normal\n");
	scanf("%d",&input);
	if (input ==1) //PINNED MEM
	{ 
		printf("Pinned!\n");
		SetPinned(42);
		printf("array = %p\n",&b );
    printf("arrayP = %p\n",b );
	}
	if(input==2)
	{
		SetUnified(1);
	}
	if (input==3)
	{
		if(!(b = (long long*)malloc(ss*sizeof(unsigned long long))))return 0;
	}

  for (long i = 0; i<m; i++){ //iterate across to next column
    for (long j =0; j<n; j++){ //iterate downwards rows.
    long k = i * n +j;
      b[k] = mat[j][i];
    }
  }
  return 1;
}
//******************************************************************

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
unsigned long long GetN(){
	return N;
}
unsigned long long getRow(){
	return aRow;
}
unsigned long long getCol(){
	return bCol;
}
dim3 GetGrid(){
	return GRID;
}
dim3 GetBlock(){
	return BLOCK;
}