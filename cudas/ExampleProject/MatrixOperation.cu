/**-------------------------------------------------------------------------------
Program to multiply 2 matrices together.
-------------------------------------------------------------------------------**/
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "MatrixOperation.h"
#include <stdio.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define MAX (x,y) ((x>y)?x:y)
#define BLOCKSIZE 32
//-------------------------------------------------------------------------------

__global__ void multiplication(int *A, int* B, int *C, int N){
   int ROW = blockIdx.y*blockDim.y+threadIdx.y;
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   
   int sum = 0;

   //if (STRIDE){}

   else if (ROW < N && COL < N){

   	for (int i = 0; i < N; ++i)
   	{
   		sum += A[ROW*N+COL] * B[ROW * N + COL];
   	}
   	
   	C[ROW*N*COL] = sum;
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

void MatrixOperation(int* aC, int* bC, int* cC, int width1, int height1, int width2,
	int height2, int*a,int*b,int*c){

	int g = sizeof(int*);
	int gg = sizeof(int**);
	printf("sizeof int*= %d, sizeof int** = %d\n",g,gg);

	int size1 = width1 * height1 * sizeof(int); //matrixA
	int size2 = width2 * height2 * sizeof(int); //matrixB

	printf("Size1 = %d\n",size1);
	printf("Height1 = %d\n",height1);
	//SetupDim(width1,height1,width2,height2);
	//if size > gridsize{}
	//else:
	cudaMalloc((void**)&aC, size1);
	cudaMalloc((void**)&bC,size2);
	cudaMalloc((void**)&cC, size2);

	int gRow = ((MAX(height2,height1)) + BLOCKSIZE - 1)/BLOCKSIZE; //5000 + 31 / 32
	int gCol = ((MAX(width2,width1)) + BLOCKSIZE - 1)/BLOCKSIZE; //5000 + 31 / 32

	dim3 dimGrid(gRow,gCol);//Number of blocks in the grid
	dim3 dimBlock (BLOCKSIZE,BLOCKSIZE); //32*32 threads per block. = 1024; Studies suggest this isn't always the most optimal.

	//TOTAL THREADS = row * col * blocksize * blocksize

	gpuErrchk(cudaMemcpy(aC,a,size1,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(bC,b,size2,cudaMemcpyHostToDevice));

	multiplication<<<dimGrid,dimBlock>>>(aC,bC,cC,height1);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(c,cC,size1,cudaMemcpyDeviceToHost));

}

void SetupDim (int width1, int height1, int width2, int height2){

	//if (compute >= 2.x) then{
	
	// if(gridDim.x*blockDim.x < size1){
	// 	STRIDE = 42; // we tell the kernel to use a stride method of multiplication.
	//	}
	//}
		//else if (compute >=3.x){}
		//else if (compute < 2.x){}
}