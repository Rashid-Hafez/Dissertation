#include<time.h>
#include<stdio.h>
#include<cuda.h>
#include <cuda_runtime.h>
#define N 30
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Global function is able to be accessed by GPU and CPU
__global__ void vectorAdd(int *a,int *b,int *c){
        if(blockIdx.x<N){
                c[blockIdx.x] = a[blockIdx.x]+b[blockIdx.x];
        }
}
void randomInit(int* data, int size)
{
    for (int i = 0; i < size; ++i){
        data[i] = rand()% (1000 + 1 - 1) + 1;
        	if(i<5){
			printf("\n%d",data[i]);
		}
        }
}

int main(){

printf("hello\n");

int size = N *sizeof(int);

int *a, *b, *c; //host vectors
int *aC,*bC,*cC;//cuda vectors
struct timespec start,stop;

        printf("\n Code to add vectors A and B");
        cudaMalloc((void**)&aC, size);
        cudaMalloc((void**)&bC,size);
        cudaMalloc((void**)&cC, size);

        a= (int *)malloc(size);
        b = (int *)malloc(size);
        c=(int *)malloc(size);
        randomInit(a,N); randomInit(b,N);

        gpuErrchk(cudaMemcpy(aC,a,size,cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(bC,b,size,cudaMemcpyHostToDevice));
	
	clock_gettime(CLOCK_REALTIME,&start);
	//Create kernel of N blocks holding 1 threads
        vectorAdd<<<N,1>>>(aC,bC,cC);//can do <<<N,1>>> for parralel
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());
        
	gpuErrchk(cudaMemcpy(c,cC,size,cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME,&stop);
	printf("\n printing 20 results of C");
                for(int i=0;i<20;i++){
                        printf("\n%d",c[i]);
                }

        printf("\n freeing all vectors from memory");
        free(a); free(b); free(c);
        cudaFree(aC); cudaFree(bC); cudaFree(cC);//changed to cuda free
return 0;
}
