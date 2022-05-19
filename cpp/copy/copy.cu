#include <stdio.h>
#include <math.h>

// Size of array
#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 512 //default
#endif

__global__ void copy_vector(float *a, float *c)
{
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_idx < ARRAY_SIZE){
        c[thread_idx] = a[thread_idx];
    }
}


int main()
{
    size_t bytes = ARRAY_SIZE*sizeof(float);

    float *A = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);

    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    for(int i=0; i<ARRAY_SIZE; i++)
    {
        A[i] = 1.0;
    }

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

    int thr_per_blk = 512;
    // Effectively an ceil
    // https://stackoverflow.com/questions/2422712/rounding-integer-division-instead-of-truncating  
    int blk_in_grid = (ARRAY_SIZE + (thr_per_blk -1))/thr_per_blk;

    dim3 grid(blk_in_grid, 1, 1);
    dim3 block(thr_per_blk, 1, 1);

    copy_vector<<< grid, block >>>(d_A, d_C);

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<ARRAY_SIZE; i++)
    {
        if(float(C[i]) != float(A[i]))
        {
          printf("\nError: value of C[%d] = %f instead of A[i]=%f\n\n", i, C[i], A[i]);
            exit(-1);
        }
    }

    free(A);
    free(C);
    cudaFree(d_A);
    cudaFree(d_C);

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", ARRAY_SIZE);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
