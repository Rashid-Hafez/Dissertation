/**

TODO: 
------------------------
Async Stream
Shared Mem/Tiling
Column based
Matrix Struct
MallocPitched
2d access in kernel
------------------------
Notes:
-------------------------------------------
nvcc -o MatrixExample MatrixExample.cu MatrixOperation.cu
cuda-memcheck MatrixExample

CANT READ AND WRITE ON SAME LINE. MUST USE SYNCTHREADS... 
so array[idx] = array [idx+1] WRONG!!!

   //must do this: temp = array[idx+1];
   //syncthreads()
   //array[idx] = temp;
   //syncthreads();
  // COALESE thrreads not stride, if threads acess farr apart. 
  //put each far apart in 1 array and process it so they all access close, then you can put them babck in the original array.

  @ References:
  - Cuda By Example
  - HP
  - Stencil Kernel

  - export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/Developer/NVIDIA/CUDA-10.0/lib
  - export PATH=/Developer/NVIDIA/CUDA-10.0/bin${PATH:+:${PATH}}
**/

/**
Example of main class doing opertations on matrices. This class takes a premade matrces and converts them to 1D array for GPU operations.
**/
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cuda.h>
#include <cuda_runtime.h>
#include "ISplit.h"

/////////////// MACROS and GLOBALS: //////////////
#define N 200000000
#define BLOCK_SIZE 32
long gMem; int gSize[3]; int wSize; int TPB;//max threads per block
const int tenMB = 10000000;
/////////////////////////////

typedef struct{
  
  unsigned long long M;//size
  unsigned int p; //partition
  unsigned int overflow; //overflow
  float * vec;
}VECTOR;

//////////////////////////////////////////////////////////
int CheckR(long long* a1, long long* b1, unsigned long long nm, long long* c);
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
  cudaEvent_t startP,stopP;
  float timeP;

  gpuErrchk(cudaEventRecord(startP,0));
  pp = getProp();
  gMem = pp.totalGlobalMem;
  gSize[0] = pp.maxGridSize[0]; gSize[1] = pp.maxGridSize[1]; gSize[2] = pp.maxGridSize[2];
  wSize = pp.warpSize;
  TPB = pp.maxThreadsPerBlock;
  printf("total Global mem: %ld\n", gMem);
  printf("maxGridSize= %d,%d,%d \n",gSize[0],gSize[1],gSize[2]);
  printf("Warp Size: %d\n", wSize);
  printf(" TPB: %d\n", TPB);
//-----------------------------------------------------------
  srand(356);

  printf("Initialised\n");
  printf("Creating Template vector\n");

  VECTOR v;
  v.M = N;
  v.p =2;
  v.overflow = 0;

  float * c;
  unsigned long byteSize = (N*sizeof(float));
  //Host
  gpuErrchk(cudaHostAlloc((void**)&v.vec,((v.M)*sizeof(unsigned long long)),cudaHostAllocDefault));
  gpuErrchk(cudaHostAlloc((void**)&c,((v.M)*sizeof(unsigned long long)),cudaHostAllocDefault));

  randomInit(v.vec);

  printf("Size of vec= %lu \n", byteSize);

  printf("----------------Split up vector-------------------------\n");

  /*------------Basic Generic CUDA Setup------------------- */

  unsigned long long Nn;

  while((byteSize/v.p)>(gMem-tenMB)){
    v.p += 2;
    Nn = v.M/v.p;
    v.overflow = v.M%v.p;
  }

  dim3 BLOCK(BLOCK_SIZE);
  unsigned long long Ns = Nn+BLOCK.x-1/BLOCK.x;
  dim3 GRID(Ns);

  printf("GRID(%d,%d,%d), BLOCK(%d,%d,%d)\n",GRID.x,GRID.y,GRID.z,BLOCK.x,BLOCK.y,BLOCK.z);

  cudaStream_t stream0;
  cudaEvent_t start,stop;
  float time;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk( cudaStreamCreate( &stream0 ) );
  cudaDeviceSynchronize();
  //Timer START LETS GOOO!
  gpuErrchk(cudaEventRecord(start,0));
  //malloc
  float * aC;
  gpuErrchk(cudaMalloc((void**)&aC, (Nn*sizeof( unsigned long long))));

//Start Loop 
for (unsigned long long i = 0; i <= v.M-v.overflow; i+=Nn){

    gpuErrchk(cudaMemcpyAsync(aC,v.vec+i,(Nn*sizeof(unsigned long long)),cudaMemcpyHostToDevice,stream0));

    Incr<<<GRID,BLOCK,0,stream0>>>(aC,Nn,i);

    gpuErrchk(cudaMemcpyAsync(c+i,aC,(Nn*sizeof(unsigned long long)),cudaMemcpyDeviceToHost,stream0)); //i = N;
  }
  if (v.overflow)
  {
    gpuErrchk(cudaMemcpyAsync(aC,v.vec+(v.M-v.overflow),(v.overflow*sizeof(unsigned long long)),cudaMemcpyHostToDevice,stream0));
    Incr<<<GRID,BLOCK,0,stream0>>>(aC,v.overflow,v.overflow);
    gpuErrchk(cudaMemcpyAsync(c+(v.M-v.overflow),aC,(v.overflow*sizeof(unsigned long long)),cudaMemcpyDeviceToHost,stream0));
  }

    gpuErrchk(cudaStreamSynchronize(stream0)); // Tell CPU to hold his horses and wait
    cudaDeviceSynchronize();
    gpuErrchk(cudaEventRecord(stop,0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&time, start, stop));

    printf("Time Taken: %3.1f ms \n",time);
    gpuErrchk(cudaStreamDestroy(stream0));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

  printf("\n freeing all vectors from memory\n");

  //CheckI(v.vec,v.M,c);

  gpuErrchk( cudaFreeHost( v.vec ) );
  gpuErrchk( cudaFree( aC ) );

  gpuErrchk(cudaEventRecord(stopP,0));
  gpuErrchk(cudaEventSynchronize(stopP));
  gpuErrchk(cudaEventElapsedTime(&timeP, startP, stopP));
  printf("Time Taken for whole program: %3.1f ms \n",timeP);

  return 0;
}

int CheckI(float * vv, unsigned long s, float *&c){
  
  for (int i = 0; i <=s ; ++i)
  {
    vv[i]+=1;

    if (vv[i]!=c[i])
    {
      printf("vv[%d]= %f, but c = %f\n",i,vv[i],c[i]);
    	return(0);
    }
  }
  return (42);
}

/**
  Verify if multiplication output is correct for dot product of vector
**/
int CheckR(long long* a1, long long* b1, unsigned long long nm, long long* c){

  long long sum = 0.0f;

  printf("C = %d\n",c[0]);

  for (int i = 0; i < nm; ++i)
  {
    sum += a1[i] * b1[i];
  }

  printf("Sum = %llu \n",sum);

  return (0);
}
