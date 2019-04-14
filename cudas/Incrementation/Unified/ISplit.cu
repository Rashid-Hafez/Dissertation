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
__global__ void Incr(float * vec, float* cc, unsigned long n, unsigned long long it){

long long x = (long long)threadIdx.x + (long long)blockIdx.x * (long long)blockDim.x;
long long y = (long long)threadIdx.y + (long long)blockIdx.y * (long long)blockDim.y;
long long offset = x + y * (long long)blockDim.x * (long long)gridDim.x; //works for any size and anything

  	if(offset<=n){
  	   	cc[offset] = vec[offset] * 3.3f;
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
