#include<cuda.h>
#include <cuda_runtime.h>

__global__ void multiplication(float *A, float* B, float *C, int N){
   int ROW = blockIdx.y*blockDim.y+threadIdx.y;
   int COL = blockIdx.x*blockDim.x+threadIdx.x;
   
   if (ROV < N && COL < N){
    C[] = A[] * B[];
   }
}
