#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include <cuda_runtime.h>

#ifndef MATRIX_WIDTH
    #define MATRIX_WIDTH 100 //default
#endif

#ifndef BLOCK_WIDTH
    #define BLOCK_WIDTH 16 //default
#endif


__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n){
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;

    if ((Row < n) && Col < n){
        float PValue = 0;
        for (int k = 0; k < n; k++){
            PValue += a[Row*n+k]*b[k*n+Col];
        }
        c[Row*n+Col] = PValue;
    }
}


int main()
{
    float *a, *b, *c;

    int n = MATRIX_WIDTH;
    int size = n * n * sizeof(float);

    a = (float*)malloc(size); 
    b = (float*)malloc(size); 
    c = (float*)malloc(size); 

    srand(0);


    for(int i=0; i<MATRIX_WIDTH*MATRIX_WIDTH; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    float *cuda_a, *cuda_b, *cuda_c;


    cudaMalloc((void **)&cuda_a, size);
    cudaMalloc((void **)&cuda_b, size);
    cudaMalloc((void **)&cuda_c, size);

    cudaMemcpy(cuda_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, size, cudaMemcpyHostToDevice);


    int NumBlocks = n/BLOCK_WIDTH;
    if(n % BLOCK_WIDTH) NumBlocks++;
    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    matMultCUDA << < dimGrid, dimBlock >> >(cuda_a , cuda_b , cuda_c , n);

    cudaMemcpy(c, cuda_c, size, cudaMemcpyDeviceToHost);

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    return 0;
}
