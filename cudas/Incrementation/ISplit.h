/*
	Rashid Hafez

*/
#include "ISplit.h"

dim3 GRID; 
dim3 BLOCK;
static cudaDeviceProp PROPS;

/******************************
  Increment Kernel
*******************************/
__global__ void Incr(float * aC, unsigned long n, unsigned long long it){

int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x; //works for any size and anything

  	if(offset<=n){
  	   	aC[offset]++;
  	 }
}


void ISplit(float * & arr, unsigned long sz, cudaDeviceProp* prop){
	
}

void setProp(int d){
	gpuErrchk(cudaSetDevice(d));
	gpuErrchk(cudaGetDeviceProperties(&PROPS,d));
}
cudaDeviceProp getProp(){
	return(PROPS);
}
nces:
  - Cuda By Example
  - HP
  - Stencil Kernel

  - export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/Developer/NVIDIA/CUDA-10.0/lib
  - export PATH=/Developer/NVIDIA/CUDA-10.0/bin${PATH:+:${PATH}}
**/

/**
Example of main class doing opertations on matrices. This class takes a pre