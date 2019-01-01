#ifndef MATRIXOPERATION_H_   /* Include guard */
#define MATRIXOPERATION_H_

void MatrixOperation(int x);  /* An example function declaration */
__global__ void multiplication(int *A, int* B, int *C, int N);
int STRIDE;
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void MatrixOperation(int* aC, int* bC, int* cC, int width1, int height1, int width2, int height2, int*a,int*b,int*c);
void SetupDim (int width1, int height1, int width2, int height2);

#endif // MATRIXOPERATION_H_
