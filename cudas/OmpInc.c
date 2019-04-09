#include <stdio.h>      /* printf, NULL */  
#include <stdlib.h>     /* srand, rand */  
#include <omp.h>   
/////////////// MACROS and GLOBALS: //////////////  
#define N 200000000 
/////////////////////////////  
  
typedef struct{  
  unsigned long long M;//size  
  unsigned int p; //partition  
  unsigned int overflow; //overflow  
  float * vec;  
}VECTOR;  
  
//////////////////////////////////////////////////////////  
void CheckI(float *vv, float *c,long long s);  
  
void randomInit(float *data) 
{  
  #pragma unroll 
    for (int i = 0; i <= N; i++){  
        data[i] = rand()% (1000 + 1 - 1) + 1;  
        }  
}  
//////////////////////////////////////////////////////////  
  
///////////////////////////////////////////////////////////////////////////////////////  
int main(){  
 
  double start_time = omp_get_wtime(); 
  srand(356);  
  
  printf("Initialised\n");  
  
  VECTOR v;  
  v.M = N;  
  v.p =2;  
  v.overflow = 0;  
  

  unsigned long byteSize = (N*sizeof(unsigned long long));  
 
 /////  Malloc Array ////// 

  v.vec = malloc(v.M * sizeof(long long)); 
  randomInit(v.vec);  
  
  printf("Size of vec= %lu \n", byteSize);  
   
  double pTime = omp_get_wtime(); 
  printf("v[1000000] = %f\n", v.vec[1000000]); 
  
  float * cc = (float*) malloc(v.M * sizeof(long long));
  CheckI(v.vec,cc,v.M); 
  double time2 = omp_get_wtime()-pTime;
  printf("Parallel Time = %6f \n",time2); 
  printf("c[1000000] = %f\n", cc[1000000]); 
  free(v.vec); 
  free(cc); 
  double time1 = omp_get_wtime() - start_time; 
  
  printf("Whole Time = %6f \n",time1);

  return(0);
} 
 
void CheckI(float * vv, float*c,long long s){  
// determine how many elements each process will work on 
   //c[900000] = 19.0f;
   //printf("created C\n");
   //printf("c[9000000] = %f\n",c[900000]);
omp_set_num_threads(32); 
//determine how many elements each process will work on 
#pragma omp parallel 
  { 
  long long  i, id, numthread = 0; 
  long long size = s; 
  id = (long long) omp_get_thread_num(); //gets the current thread number 
  numthread = (long long) omp_get_num_threads(); 

  for(i = id; i<=size; i+=numthread)  
    {    
      c[i] = vv[i]*3.3f;
    }
  } 
}