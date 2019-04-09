/*
	Transpose.
  
  Have 4 matrices. 

  M - original
  A - pinned placeholder for partition to send *TO* kernel
  C - pinned placeholder for partition result *FROM* kernel
  Out - result

*/
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cuda.h>
#include <cuda_runtime.h>
#include "TSplit.h"

/////////////// MACROS and GLOBALS: //////////////
#define N 50000000
#define BLOCK_SIZE 32
#define oneGB 1000000000
#define SubsMat 4
long gMem; int gSize[3]; int wSize; int TPB;//max threads per block
/////////////////////////////
typedef struct
{
  unsigned long long row;
  unsigned long long col;
  unsigned long long size;
  int p; //partition
  int stride;
  int overflow;
  long long * mat;
}MATRIX;

//////////////////////////////////////////////////////////
MATRIX Cut(unsigned long long *M,unsigned long long rows, unsigned long long cols);
void Glue(unsigned long long *M);
int CheckT(unsigned long long * &M, unsigned long long * &C);

void randomInit(float* &data, unsigned long long size)
{
  #pragma unroll
    for (int i = 0; i <= size; i++){
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
  printf("Creating Template Matrix\n");

  MATRIX matM; //original
  MATRIX matR; //final full result

  M.row = N; M.col = N; M.size = M.row*M.col;

  unsigned long byteSize = ((M.row*M.col)*sizeof(unsigned long long));
  M.matM = (unsigned long long *)malloc(M.row*M.col*sizeof(unsigned long long));

  randomInit(M.mat,(M.row*M.col)); //INITIALIZE
  printf("Size of Mat= %lu \n", byteSize);

  printf("----------------Split up Mat-------------------------\n");

  unsigned long long Nn = ceil(M.size / M.p);
  unsigned long long bt = (long long)byteSize/M.p;
  unsigned long long mem = (long long) (gMem-oneGB);
  printf("Nn=%llu, bt=%llu, mem=%llu",Nn,bt,mem);

  long long Max = (M.row*M.col)/SubsMat;
  Max -= Max % M.row;
  printf("Max = %llu\n", Max);



  // while((bt*2)>mem){

  //   M.p += 2;
  //   bt = (long long)byteSize/M.p;
  //   Nn = M.size / M.p;
  //   M.overflow = M.size%M.p;
  // }
  int sub = SubsMat / 2; //4 sub matrices, 2 on the top, 2 on bottom
  unsigned long long SubRow1 = M.row/sub; //how many rows in the first matrix
  unsigned long long SubRow2 = M.row - SubRow1; // how many rows in the second submatrix

  unsigned long long SubCol2 = M.col/sub; //how many rows in the first matrix 
  unsigned long long SubCol1 = M.col - SubCol1; //how many rows in the second matrix

/*------------------------ C U T -------------------------------------*/
  for(unsigned long long y = 0; y <= sub; y++){
      // if(overflowA == 0 && y == SubsMat){
      //   break;
      // }

      MATRIX temp;

      temp.mat = (long long*) malloc( sizeof(long long)*SubRow1 * SubCol1 );

      for(int j = 0; j <= SubRow1; j++){
        for(int x = 0; x <= SubCol1; x++){
          if(y * SubRow1 + j < m){
            temp.mat[j * k + x] = M.mat[j*m.Row+(y*SubCol1+x)];
          }else{
            temp.mat[j * k + x] = 0;
          }
        }
      }
    }
/*------------------------ Cut END -------------------------------------*/

/*------------Basic Generic CUDA Setup------------------- */
  dim3 BLOCK(BLOCK_SIZE, BLOCK_SIZE);
  dim3 GRID(Nn+BLOCK.x-1/BLOCK.x); //flexible size of grid for multiple of datasize

  printf("GRID(%lu,%d,%d), BLOCK(%d,%d,%d)\n",GRID.x,GRID.y,GRID.z,BLOCK.x,BLOCK.y,BLOCK.z);
  printf("partition = %lu\n",M.p);
  printf("overflow= %d \n",M.overflow);

  cudaEvent_t start,stop;
  float time;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  //Timer START LETS GOOO!
  gpuErrchk(cudaEventRecord(start,0));
  //malloc
  long long *inDev;
  long long *outDev;
  gpuErrchk(cudaMalloc((void**)&inDev, (Nn*sizeof( unsigned long long))));
  gpuErrchk(cudaMalloc((void**)&outDev, (Nn*sizeof( unsigned long long))));

  gpuErrchk(cudaMemcpy(inDev,matA.mat,(Nn*sizeof(unsigned long long)),cudaMemcpyHostToDevice)); //send

  Transp<<<GRID,BLOCK>>>(outDev,inDev,matA.row,matA.col); //compute

  gpuErrchk(cudaMemcpy(c+i,aC,(Nn*sizeof(unsigned long long)),cudaMemcpyDeviceToHost)); //recieve

  cudaDeviceSynchronize();
  gpuErrchk(cudaEventRecord(stop,0));
  gpuErrchk(cudaEventSynchronize(stop));

  checkT(matM.mat,matR.mat,matM.row,matM.col); // check result

  }//main end

  int checkT(long long *&Mat, long long *&result, unsigned long long row, unsigned long long col){

    for (int i = 0; i < m; ++i ) //1
    {
       for (int j = 0; j < n; ++j ) //0
       {
          // Index in the original matrix.
          int index1 = i*n+j; //

          // Index in the transpose matrix.
          int index2 = j*m+i;

          if(result[index2] == Mat[index1]){

          }
       }
    }
  }

  /**
  * Split the Matrix up accordingly
  **/
  MATRIX Cut(unsigned long long rows, unsigned long long cols){

  }

  /**
  * Puts back in correct position
  **/
  void Glue(){
    
  }
