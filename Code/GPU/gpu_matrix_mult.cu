#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void gpu_matrix_mult(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    float *A, *B, *C;

    // Allocate Unified Memory (accessible from CPU/GPU)
    cudaMallocManaged(&A, N * N * sizeof(float));
    cudaMallocManaged(&B, N * N * sizeof(float));
    cudaMallocManaged(&C, N * N * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // Configure kernel dimensions (16x16 threads per block)
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    gpu_matrix_mult<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();  // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    printf("GPU Time: %.3f ms\n", time);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}