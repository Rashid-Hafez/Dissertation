#include <stdio.h>      /* printf, NULL */ 
#include <stdlib.h>     /* srand, rand */ 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include "ISplit.h" 
 
/////////////// MACROS and GLOBALS: ////////////// 
#define N 400000000 
#define BLOCK_SIZE 32 
#define oneGB 100000000 
long gMem; int gSize[3]; int wSize; int TPB;//max threads per block 
///////////////////////////// 
 
typedef struct{ 
   
  unsigned long long M;//size 
  unsigned int p; //partition 
  unsigned int overflow; //overflow 
  float * vec; 
}VECTOR; 
 
////////////////////////////////////////////////////////// 
int CheckI(float *vv, unsigned long s, float *&c); 
 
void randomInit(float* &data) 
{ 
  #pragma unroll 
    for (int i = 0; i <= N; i++){ 
        data[i] = rand()% (1000 + 1 - 1) + 1; 
        } 
} 
////////////////////////////////////////////////////////// 
 
/////////////////////////////////////////////////////////////////////////////////////// 
int main(int argc, char** argv){ 
 
  //Setup Check// 
  int Dev = 0; 
  cudaDeviceProp pp; 
  setProp(Dev); 
  pp = getProp(); 
   
  cudaEvent_t start1,stop1;  
  float time1; 
  gpuErrchk(cudaEventCreate(&start1)); 
  gpuErrchk(cudaEventCreate(&stop1)); 
  gpuErrchk(cudaEventRecord(start1,0)); 
 
  gMem = pp.totalGlobalMem; 
  gSize[0] = pp.maxGridSize[0]; gSize[1] = pp.maxGridSize[1]; gSize[2] = pp.maxGridSize[2]; 
  wSize = pp.warpSize; 
  TPB = pp.maxThreadsPerBlock; 
//  printf("total Global mem: %ld\n", gMem); 
//  printf("maxGridSize= %d,%d,%d \n",gSize[0],gSize[1],gSize[2]); 
//  printf("Warp Size: %d\n", wSize); 
//  printf(" TPB: %d\n", TPB); 
//----------------------------------------------------------- 
  srand(356); 
 
  printf("Initialised\n"); 
//  printf("Creating Template vector\n"); 
 
  VECTOR v; 
  v.M = N; 
  v.p =2; 
  v.overflow = 0; 
 
  float * c; 
  unsigned long byteSize = (N*sizeof(long long)); 
  //Host 
  gpuErrchk(cudaHostAlloc((void**)&v.vec,((v.M)*sizeof(long long)),cudaHostAllocDefault)); 
  gpuErrchk(cudaHostAlloc((void**)&c,((v.M)*sizeof(long long)),cudaHostAllocDefault)); 
 
  randomInit(v.vec); 
 
 printf("Size of vec= %lu \n", byteSize); 
 
  /*------------Basic Generic CUDA Setup------------------- */ 
 
  unsigned long Nn = ceil(v.M / v.p); 
  unsigned long bt = (long long)byteSize/v.p; 
  unsigned long long mem = (long long) (gMem-oneGB); 
   
  while((bt*2)>mem){ 
     
    v.p += 2; 
    bt = (long long)byteSize/v.p; 
    Nn = v.M/v.p; 
    v.overflow = v.M%v.p; 
  } 

  dim3 BLOCK(BLOCK_SIZE); 
   dim3 GRID(Nn+BLOCK.x-1/BLOCK.x); 
 
  //printf("GRID(%lu,%d,%d), BLOCK(%d,%d,%d)\n",GRID.x,GRID.y,GRID.z,BLOCK.x,BLOCK.y,BLOCK.z); 
  //printf("partition = %lu\n",v.p); 
  //printf("overflow= %d \n",v.overflow); 
  cudaStream_t stream0; 
  cudaStream_t stream1; 
 
  cudaEvent_t start,stop; 
  float time; 
 
  gpuErrchk(cudaEventCreate(&start)); 
  gpuErrchk(cudaEventCreate(&stop)); 
  gpuErrchk( cudaStreamCreate( &stream0)); 
  gpuErrchk( cudaStreamCreate( &stream1)); 
  //Timer START LETS GOOO! 
  gpuErrchk(cudaEventRecord(start,0)); 
  //malloc 
  float * aC; 
  float * aC1; 
  gpuErrchk(cudaMalloc((void**)&aC, (Nn*sizeof(long long)))); 
  gpuErrchk(cudaMalloc((void**)&aC1, (Nn*sizeof(long long)))); 
 
//----------------------START LOOP--------------------------------); 
 
for (unsigned long long i = 0; i < v.M-v.overflow; i+=Nn*2){ //Nn*2 because 2 streams 
 
    gpuErrchk(cudaMemcpyAsync(aC,v.vec+i,(Nn*sizeof(long long)),cudaMemcpyHostToDevice,stream0)); 
    gpuErrchk(cudaMemcpyAsync(aC1,v.vec+(i+Nn),(Nn*sizeof(long long)),cudaMemcpyHostToDevice,stream1));  
     
    Incr<<<GRID,BLOCK,0,stream0>>>(aC,Nn,i); 
    Incr<<<GRID,BLOCK,0,stream1>>>(aC1,Nn,i); 
     
    gpuErrchk(cudaMemcpyAsync(c+i,aC,(Nn*sizeof(long long)),cudaMemcpyDeviceToHost,stream0)); //i = N; 
    gpuErrchk(cudaMemcpyAsync(c+(i+Nn),aC1,(Nn*sizeof(long long)),cudaMemcpyDeviceToHost,stream1)); 
  } 
   
    if (v.overflow) 
    { 
      gpuErrchk(cudaMemcpyAsync(aC,v.vec+(v.M-v.overflow),(v.overflow*sizeof(long long)),cudaMemcpyHostToDevice,stream1)); 
      Incr<<<GRID,BLOCK,0,stream1>>>(aC,v.overflow,v.overflow); 
      gpuErrchk(cudaMemcpyAsync(c+(v.M-v.overflow),aC,(v.overflow*sizeof(long long)),cudaMemcpyDeviceToHost,stream1)); 
    } 

    gpuErrchk(cudaStreamSynchronize(stream0)); // Tell CPU to hold his horses and wait 
    gpuErrchk(cudaStreamSynchronize(stream1)); // Tell CPU to hold his horses and wait 
    cudaDeviceSynchronize(); 
    gpuErrchk(cudaEventRecord(stop,0)); 
    gpuErrchk(cudaEventSynchronize(stop)); 
    gpuErrchk(cudaEventElapsedTime(&time, start, stop)); 
 
    gpuErrchk(cudaStreamDestroy(stream0)); 
    gpuErrchk(cudaStreamDestroy(stream1)); 
    gpuErrchk(cudaEventDestroy(start)); 
    gpuErrchk(cudaEventDestroy(stop)); 
    printf("2 Stream\n"); 
 
    //CheckI(v.vec,v.M,c); 
     
    gpuErrchk( cudaFreeHost( v.vec ) ); 
    gpuErrchk( cudaFree( aC ) ); 
    gpuErrchk( cudaFree( aC1 ) ); 
 
    gpuErrchk(cudaEventRecord(stop1,0));  
    gpuErrchk(cudaEventSynchronize(stop1)); 
    gpuErrchk(cudaEventElapsedTime(&time1, start1, stop1)); 
    gpuErrchk(cudaEventDestroy(stop1));   
    gpuErrchk(cudaEventDestroy(start1)); 
    printf("Parallel Time Taken: %3.1f ms \n",time); 
    time1 +=time; 
    printf("Full Time Taken: %6f seconds \n",time1/1000.0000); 
return 0; 
} 
 
int CheckI(float * vv, unsigned long s, float *&c){ 
   
  for (int i = 0; i <=s ; ++i) 
  { 
    vv[i]*=3.3; 
 
    if (vv[i]!=c[i]) 
    { 
      printf("vv[%d]= %f, but c = %f\n",i,vv[i],c[i]); 
      return(42); 
    } 
  } 
  return (0); 
}