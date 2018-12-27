/* ttys000
Hanis-MBP:~ rashid$ ls
Applications                      Downloads
Auxiliary                         Library
Bootstrap.cmk                     Makefile
CMake.DeveloperReference.HTML.qs  Modules
CMake.Dialogs.QtGUI.qs            Movies
CMake.Documentation.SphinxHTML.qs Music
CMake.qs                          NVIDIA_CUDA-10.0_Samples
CMakeCPackOptions.cmake           Pictures
CMakeCache.txt                    Public
CMakeFiles                        Rashid-Hafez.github.io.git
CPackConfig.cmake                 Source
CPackSourceConfig.cmake           Testing
CTestCustom.cmake                 Tests
CTestScript.cmake                 Utilities
CTestTestfile.cmake               bin
DartConfiguration.tcl             cmake_install.cmake
Desktop                           cmake_uninstall.cmake
Desktop keystore.jks              eclipse
Documents                         eclipse-workspace
Hanis-MBP:~ rashid$ ssh -Y rh1@jove.macs.hw.ac.uk
Last login: Wed Dec 26 17:10:20 2018 from 0541b029.skybroadband.com
jove:~$ ssh -Y rhafez@robotarium.hw.ac.uk
Last login: Wed Dec 26 17:10:29 2018 from 137.195.27.15
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                             ======================
                             Heriot-Watt University
                               Robotarium Cluster
                             ======================
                               Based on RHEL 7.3

==> Submitting Jobs to the SLURM batch management system
 - Interactive:
   srun -p defq <EXEC> <ARGS>

 - In the Background:
   sbatch -p defq <SCRIPT> <ARGS>

 - Interactive Session:
   srun --pty /usr/bin/bash
 
 * Check current state of Queues (Partitions):
   squeue
 * Check current state of Nodes:
   sinfo

==> Software Modules
 - Most software is available through Environment Modules:
   module avail            - show available software modules
   module list             - list loaded software modules
   module add <MODULE>     - add a module to your environment (this session)
   module initadd <MODULE> - configure module to be loaded at every login

==> Further Documentation and Support:
 - Wiki: http://www.macs.hw.ac.uk/~hv15/robotarium/
 - Slack: https://robotariumcluster.slack.com/

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

gcc/6.3.0(15):ERROR:105: Unable to locate a modulefile for 'nsight'
[rhafez@robotarium ~]$ cd Projects
[rhafez@robotarium Projects]$ cd c
*/

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
