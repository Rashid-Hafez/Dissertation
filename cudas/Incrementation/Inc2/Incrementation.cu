#include <time.h>       /* time */ 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include "ISplit.h" 
 
/////////////// MACROS and GLOBALS: ////////////// 
#define N 100000000 
#define oneGB 100000000 
#define BLOCK_SIZE 32 
unsigned long long gMem; int gSize[3]; int wSize; int TPB;//max threads per block 
///////////////////////////// 
 
typedef struct{ 
   
  unsigned long long M;//size 
  size_t p; //partition 
  unsigned int overflow;  
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
   
  //timer 
  cudaEvent_t start1,stop1;  
  float time1;
  gpuErrchk(cudaEventCreate(&start1));  
  gpuErrchk(cudaEventCreate(&stop1));  
  //Timer START LETS GOOO!  
  gpuErrchk(cudaEventRecord(start1,0));  

  gMem = pp.totalGlobalMem; 
  gSize[0] = pp.maxGridSize[0]; gSize[1] = pp.maxGridSize[1]; gSize[2] = pp.maxGridSize[2]; 
  wSize = pp.warpSize; 
  TPB = pp.maxThreadsPerBlock; 
  //printf("total Global mem: %ld\n", gMem); 
  //printf("maxGridSize= %d,%d,%d \n",gSize[0],gSize[1],gSize[2]); 
  //printf("Warp Size: %d\n", wSize); 
  //printf(" TPB: %d\n", TPB); 
//----------------------------------------------------------- 
  srand(356); 

  printf("Initialised\n"); 
  //printf("Creating Template Matrix\n"); 

  VECTOR v; 
  v.M = N; 
  v.p = 2; 
  v.overflow = 0; 
  float * c; 
  unsigned long byteSize = (N*sizeof(unsigned long long)); 
  //Host 
  gpuErrchk(cudaHostAlloc((void**)&v.vec,((v.M)*sizeof(unsigned long long)),cudaHostAllocDefault)); 
  gpuErrchk(cudaHostAlloc((void**)&c,((v.M)*sizeof(unsigned long long)),cudaHostAllocDefault)); 

  randomInit(v.vec); 

  printf("Size of vec= %lu \n", byteSize); 

  printf("----------------Split up vector-------------------------\n"); 

  /*------------Basic Generic CUDA Setup------------------- */ 
  float * aC; 
   
  unsigned long long Nn = ceil(v.M / v.p); 
  unsigned long long bt = (long long)byteSize/v.p; 
  unsigned long long mem = (long long) (gMem-oneGB); 
  //printf("Nn=%llu, bt=%llu, mem=%llu",Nn,bt,mem); 
  while(bt>mem){ 
     
    v.p += 2; 
    bt = (long long)byteSize/v.p; 
    Nn = v.M/v.p;
    v.overflow = v.M%v.p; 
  } 

  cudaEvent_t start,stop; 
  float time; 

  gpuErrchk(cudaEventCreate(&start)); 
  gpuErrchk(cudaEventCreate(&stop)); 
  //Timer START LETS GOOO! 
  gpuErrchk(cudaEventRecord(start,0)); 
  //malloc 
  gpuErrchk(cudaMalloc((void**)&aC, (Nn*sizeof( unsigned long long)))); 

  dim3 BLOCK(BLOCK_SIZE); 
  dim3 GRID((Nn+BLOCK.x-1/BLOCK.x)); 

 // printf("GRID(%d,%d,%d), BLOCK(%d,%d,%d)\n",GRID.x,GRID.y,GRID.z,BLOCK.x,BLOCK.y,BLOCK.z); 

//----------------------START LOOP--------------------------------; 

for (unsigned long long i = 0; i < v.M; i+=Nn){ 

    gpuErrchk(cudaMemcpy(aC,v.vec+i,(Nn*sizeof(unsigned long long)),cudaMemcpyHostToDevice)); 

    Incr<<<GRID,BLOCK,0>>>(aC,Nn,i); 

    gpuErrchk(cudaMemcpy(c+i,aC,(Nn*sizeof(unsigned long long)),cudaMemcpyDeviceToHost)); //i = N; 
  }   
  if (v.overflow) 
  { 
    gpuErrchk(cudaMemcpy(aC,v.vec+(v.M-v.overflow),(v.overflow*sizeof(unsigned long long)),cudaMemcpyHostToDevice)); 
    Incr<<<GRID,BLOCK,0>>>(aC,v.overflow,v.overflow); 
    gpuErrchk(cudaMemcpy(c+(v.M-v.overflow),aC,(v.overflow*sizeof(unsigned long long)),cudaMemcpyDeviceToHost)); 
  } 

//----------------------END LOOP-------------------------------- 
    cudaDeviceSynchronize(); 
    gpuErrchk(cudaEventRecord(stop,0)); 
    gpuErrchk(cudaEventSynchronize(stop)); 
    gpuErrchk(cudaEventElapsedTime(&time, start, stop)); 

    printf("Time Taken: %3.1f ms/n \n",time); 
    gpuErrchk(cudaEventDestroy(start)); 
    gpuErrchk(cudaEventDestroy(stop)); 
    printf("No stream\n"); 

  printf("\n freeing all vectors from memory\n"); 
  //  CheckI(v.vec,v.M,c); 
 
  gpuErrchk( cudaFreeHost( v.vec ) ); 
  gpuErrchk( cudaFree( aC ) ); 
 
    gpuErrchk(cudaEventRecord(stop1,0));  
    gpuErrchk(cudaEventSynchronize(stop1));  
    gpuErrchk(cudaEventElapsedTime(&time1, start1, stop1));  
  
    printf("Whole Time Taken: %6f s/n \n",time1/1000);  
    gpuErrchk(cudaEventDestroy(start1));  
    gpuErrchk(cudaEventDestroy(stop1));  
 
  return 0; 
} 
//Changed kernel to *3.3 
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
  return (0); 
} 