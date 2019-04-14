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
#include <time.h>       /* time */
#include "ISplit.h"

/////////////// MACROS and GLOBALS: //////////////
#define N 400000000
#define BLOCK_SIZE 32
#define oneGB 100000000
long gMem; int gSize[3]; int wSize; int TPB;//max threads per block
int major,minor;
/////////////////////////////

typedef struct{
  
  unsigned long long M;//size
  unsigned int p; //partition
  unsigned int overflow; //overflow
  float * vec;
}VECTOR;

//////////////////////////////////////////////////////////
int CheckI(float *vv, unsigned long s, float *c);

void randomInit(float*data, unsigned long s)
{
    for (long in =0; in <= s; in++){
        data[in] = rand()% (1000 + 1 - 1) + 1;
        }
}
//////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){

  //Setup Check//
  srand(356);
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
  major = pp.major; minor = pp.minor;
  printf("Compute Capability = %d.%d\n",major,minor );
  if (major<3)
  {
    fprintf(stderr, "Whoops Sorry! This device does not support unified memory!!\n", 30);
    fprintf(stderr, "Exiting...",30);
    exit(42);
  }
//  printf("total Global mem: %ld\n", gMem);
//  printf("maxGridSize= %d,%d,%d \n",gSize[0],gSize[1],gSize[2]);
//  printf("Warp Size: %d\n", wSize);
//  printf(" TPB: %d\n", TPB);
//-----------------------------------------------------------

  printf("Initialised\n");

  VECTOR v;
  v.M = N;
  v.p =2;
  v.overflow = 0;
  float * c;
  unsigned long byteSize = (v.M*sizeof(unsigned long long));
  unsigned long Nn = ceil(v.M / v.p); //how many elements in partition
  unsigned long long bt = (long long)byteSize/v.p; //how many bytes in partition
  unsigned long long mem = (long long) (gMem-oneGB); //total memory size -1GB
  
  while((bt*2)>mem){ //while the bytes size of partition greater than the memory on gpu
    
    v.p += 2; //make 2 more partitions
    bt = (long long)byteSize/v.p;
    Nn = v.M/v.p; //size of array/partition adjust how many elements per partition
    v.overflow = v.M%v.p; //how many elements are remainder
  }

  dim3 BLOCK(BLOCK_SIZE); //set the blocksize
  dim3 GRID(Nn+BLOCK.x-1/BLOCK.x); //set the grid size to be divisible by the blocksize

  cudaStream_t stream0;
  cudaEvent_t start,stop;
  float time;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk( cudaStreamCreate(&stream0));

  gpuErrchk(cudaMallocManaged(&v.vec, Nn*sizeof(unsigned long long)));
  gpuErrchk(cudaMallocManaged(&c, Nn*sizeof(unsigned long long)));
  gpuErrchk(cudaStreamAttachMemAsync(stream0,v.vec, 0,cudaMemAttachSingle));
  gpuErrchk(cudaStreamAttachMemAsync(stream0, c, 0,cudaMemAttachSingle));
  //Timer START LETS GOOO!
  gpuErrchk(cudaEventRecord(start,0));
  
  //----------------------START LOOP--------------------------------//

for (unsigned long long i = 0; i < v.M-v.overflow; i+=Nn){

    randomInit(v.vec, Nn);
    Incr<<<GRID,BLOCK,0,stream0>>>(v.vec,c,Nn,i);
    cudaDeviceSynchronize();
  }

if (v.overflow)
{
  for (long long i = v.M-v.overflow; i < v.overflow; ++i)
  {
    c[i] = v.vec[i]*3.3;
  }
}
//----------------------END LOOP--------------------------------//

    gpuErrchk(cudaStreamSynchronize(stream0)); // Tell CPU to hold his horses and wait
    cudaDeviceSynchronize();

    gpuErrchk(cudaFree(v.vec));
    gpuErrchk(cudaFree(c));

    gpuErrchk(cudaEventRecord(stop,0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&time, start, stop));
    gpuErrchk(cudaStreamDestroy(stream0));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    gpuErrchk(cudaEventRecord(stop1,0));  
    gpuErrchk(cudaEventSynchronize(stop1)); 
    gpuErrchk(cudaEventElapsedTime(&time1, start1, stop1)); 
    gpuErrchk(cudaEventDestroy(stop1));   
    gpuErrchk(cudaEventDestroy(start1)); 
    printf("Parallel Time Taken: %6f ms \n",time); 
    printf("Full Time Taken: %6f seconds \n",time1/1000.0000); 
    printf("1 Stream\n");
  return 0;
}

int CheckI(float * vv, unsigned long s, float *c){
  
  for (long long in = 0; in <s-1 ; ++in)
  {
    vv[in]*=3.3f;

    if (vv[in]!=c[in] || c[in]==0.0f)
    {
      printf("vv[%llu]= %f, but c = %f\n",in,vv[in],c[in]);
      return(0);
    }
  }
  return (42);
}


