#ifndef ISplit_H_   /* Include guard */ 
#define ISplit_H_ 
 
#include <stdlib.h> 
#include <stdio.h>      /* printf, NULL */ 
#include <cuda.h> 
#include <cuda_runtime.h> 
 
#define BLOCKSIZE 32 
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){ 
 
   if (code != cudaSuccess) 
   { 
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
      if (abort) exit(code); 
   } 
} 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } 
 
void ISplit(float * arr, long long sz,cudaDeviceProp* prop);   
 
__global__ void Incr(float * aC, unsigned long n,unsigned long long it); 
 
void SetupDim (long long width1, long long height2,cudaDeviceProp prop); 
 
void setProp(int d); 
cudaDeviceProp getProp(); 
void SetPinned(int i); 
void SetUnified(int i); 
int GetUnified(); 
int SetUnified(); 
unsigned long long GetN(); 
unsigned long long getRow(); 
unsigned long long getCol(); 
dim3 GetGrid(); 
dim3 GetBlock(); 
 
#endif // ISPLIT_H 