#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include <cuda_runtime.h>

#ifndef MATRIX_WIDTH
    #define MATRIX_WIDTH 100 //default
#endif
#ifndef BLOCK_WIDTH
    #define BLOCK_WIDTH 32 //default
#endif

__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n){
    __shared__ float a_shared[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float b_shared[BLOCK_WIDTH][BLOCK_WIDTH];

    int b_x = blockIdx.x; 
    int b_y = blockIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    
    int Row = b_y * BLOCK_WIDTH + t_y;
    int Col = b_x * BLOCK_WIDTH + t_x;
    
    float computed_value = 0;
    for (int m = 0; m < (n+BLOCK_WIDTH-1)/BLOCK_WIDTH; m++){
        
        //copy to shared memory
        if(m*BLOCK_WIDTH + t_x < n && Row < n)
            a_shared[t_y][t_x] = a[Row * n + m*BLOCK_WIDTH + t_x];
        else
            a_shared[t_y][t_x] = 0;    
        
        if(m*BLOCK_WIDTH + t_y < n && Col < n)
            b_shared[t_y][t_x] = b[(m*BLOCK_WIDTH + t_y) * n + Col];
        else
           b_shared[t_y][t_x] = 0; 
        __syncthreads(); //sync to make sure all data is available in shared memory before computations
        for (int k = 0; k < BLOCK_WIDTH; ++k){
            computed_value += a_shared[t_y][k] * b_shared[k][t_x];
        }
        __syncthreads(); //sync to ensure all threads finished using shared memory before we move
    }
    if(Row < n && Col < n)
        c[Row * n + Col] = computed_value;
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
