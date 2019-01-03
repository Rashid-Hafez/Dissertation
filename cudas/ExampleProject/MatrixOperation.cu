/**-------------------------------------------------------------------------------
Program to multiply 2 matrices together.
-------------------------------------------------------------------------------**/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "MatrixOperation.h"
#include <stdio.h>
#define MAX(a,b) (a>b ? a:b)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
int STRIDE;
int nStreams;
long long GRIDSIZE = (long long)65535L*65535L; //Larger than this does not fit on GPU
int BLOCKSIZE =  32;
static cudaDeviceProp PROPS;
//-------------------------------------------------------------------------------

__global__ void multiplication(int *A, int* B, int *C, int N){
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; // BlockIndex * BlocksizeY + ThreadY
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   int sum = 0;

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

void MatrixOperation(int* aC, int* bC, int* cC, long width1, long height1, long width2,
	long height2, int*a,int*b,int*c, cudaDeviceProp *prop){

	if(width1 != height2){
		printf("Error, width of first mat must = height of second mat\n");
		exit(0);
	}

	long size1 = width1 * height1 * sizeof(int); //matrixA
	long size2 = width2 * height2 * sizeof(int); //matrixB

	printf("Size1 = %d\n",size1);
	printf("Height1 = %d\n",height1);
	//SetupDim(width1,height1,width2,height2);
	
	//else:
	cudaMalloc((void**)&aC, size1);
	cudaMalloc((void**)&bC,size2);
	cudaMalloc((void**)&cC, size2);

	int gRow = ceil(height2/BLOCKSIZE);
	int gCol = ceil(width1/BLOCKSIZE);
	dim3 dimGrid(gRow,gCol);//Number of blocks in the grid, will be a bit larger because of Ceil.
	dim3 dimBlock (BLOCKSIZE,BLOCKSIZE); //32*32 threads per block. = 1024; Studies suggest this isn't always the most optimal.

	//TOTAL THREADS = grow * gcol * blocksize * blocksize

	gpuErrchk(cudaMemcpy(aC,a,size1,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(bC,b,size2,cudaMemcpyHostToDevice));

	multiplication<<<dimGrid,dimBlock>>>(aC,bC,cC,height1);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(c,cC,size1,cudaMemcpyDeviceToHost));

}

void SetupDim (long width1, long height1, long width2, long height2, dim3 * grid, dim3* block, cudaDeviceProp *prop){

	if (prop.major>=2)
	{
		printf("Device compute is 2 or over\n");
		
		*grid();
		*block(BLOCKSIZE,BLOCKSIZE);
	}
	else{
		printf("Device Compute Capacity less than 2\n");
		BLOCKSIZE = 16;
		
		*block(BLOCKSIZE,BLOCKSIZE);
	}
	//if (compute >= 2.x) then{
	
	// if(gridDim.x*blockDim.x < size1){
	// 	STRIDE = 42; // we tell the kernel to use a stride method of multiplication.
	//	}
	//}
		//else if (compute >=3.x){}
		//else if (compute < 2.x){}
}

/**
Convert normal matrix to ROW MAJOR matrix, if the matrices are bigger than the GPU memory the function will use pinned memory (i.e. hostmalloc). 

a(i,j) can be flatten to 1D array b(k)
mat[0] to mat[m] = the first row, mat[m+1] = the second row. mat[2*m+1] = third row
@Param: 
  - mat : the 2D matrix to convert to 1D
  - n : amount of rows
  - m : amount of colombs in the matrix
**/
int * RowMajorMat(int** mat, long n, long m){
int * newMat;
long long ss = n*m;
 if(ss <= GRIDSIZE){
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
	if (input ==1)
	{ 
		printf("Pinned!\n");
		gpuErrchk(cudaHostAlloc((void**)&mat,ss*sizeof(int),cudaHostAllocDefault)); //Page locked
	}
	if (input == 2)
	{
		/* cudaMallocManaged() */
	}
}

void setProp(int d){
	cudaSetDevice(d);
	cudaGetDeviceProperties(&PROPS,d);
}
cudaDeviceProp getProp(){
	return(PROPS);
}