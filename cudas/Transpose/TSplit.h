#ifndef TSplit_H_   /* Include guard */
#define TSplit_H_

#include <stdlib.h>
#include <stdio.h>      /* printf, NULL */
#include <cuda.h>
#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){

   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void TSplit(float * arr, long long sz,cudaDeviceProp* prop);  

__global__ void Transp(long long *out , long long *in, unsigned long long row, unsigned long long col);

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

#endif // TSPLIT_H
