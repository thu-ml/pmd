#include "pairwise_l1.h"
#include "math.h"
#include "assert.h"

// Matrix multiplication kernel called by MatMul()
__global__ void PairwiseL1Kernel(float *A, float *B, float *C, int n, int m, int d) {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= n || col >= m) return;
    for (int e = 0; e < d; ++e)
        Cvalue += fabs(A[row * d + e] - B[col * d + e]);
    C[row * m + col] = Cvalue;
}

float *d_A = 0;
float *d_B = 0;
float *d_C = 0;

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void PairwiseL1(float *A, float *B, float *C, int n, int m, int d) {
    // Load A and B to device memory
    size_t max_size = 4000 * 4000 * sizeof(float);
    size_t size = n * d * sizeof(float);
    assert(size <= max_size);
    cudaError_t err;

    if (!d_A) {
        err = cudaMalloc(&d_A, max_size);
        //printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
        //printf("Copy A to device: %s\n",cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    size = m * d * sizeof(float);
    assert(size <= max_size);
    if (!d_B) {
        err = cudaMalloc(&d_B, max_size);
        //printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
        //printf("Copy B to device: %s\n",cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    size = n * m * sizeof(float);
    assert(size <= max_size);
    if (!d_C) {
        err = cudaMalloc(&d_C, max_size);
        //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));
    }

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);
    PairwiseL1Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n, m, d);
    err = cudaThreadSynchronize();
    //printf("Run kernel: %s\n", cudaGetErrorString(err));

    // Read C from device memory
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    //printf("Copy C off of device: %s\n",cudaGetErrorString(err));

    // Free device memory
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
}

