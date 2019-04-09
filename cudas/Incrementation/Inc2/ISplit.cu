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
 
unsigned long long x = (unsigned long long)threadIdx.x + (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x; 
unsigned long long y = (unsigned long long) threadIdx.y + (unsigned long long)blockIdx.y * (unsigned long long)blockDim.y; 
unsigned long long offset = x + y * (unsigned long long)blockDim.x * (unsigned long long)gridDim.x; //works for any size and anything
 
        if(offset<=n){ 
                aC[offset]*=3.3; 
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