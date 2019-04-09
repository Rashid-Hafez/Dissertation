/*
	Rashid Hafez

*/
#include "TSplit.h"

dim3 GRID; 
dim3 BLOCK;
static cudaDeviceProp PROPS;

/******************************
  Increment Kernel
*******************************/
__global__ void Transp(long long *out , long long *in, unsigned long long row, unsigned long long col){

unsigned long long x = (unsigned long long)threadIdx.x + (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x;
unsigned long long y = threadIdx.y + (unsigned long long)blockIdx.y * (unsigned long long)blockDim.y;
unsigned long long offset = x + y * (unsigned long long)blockDim.x * (unsigned long long)gridDim.x; //works for any size and anything

int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

int index_in  = xIndex + width * yIndex;
int index_out = yIndex + height * xIndex;
for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
{
	odata[index_out+i] = idata[index_in+i*width];
}

}


void TSplit(float * & arr, unsigned long sz, cudaDeviceProp* prop){
	
}

void setProp(int d){
	gpuErrchk(cudaSetDevice(d));
	gpuErrchk(cudaGetDeviceProperties(&PROPS,d));
}
cudaDeviceProp getProp(){
	return(PROPS);
}
