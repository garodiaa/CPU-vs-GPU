#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void cpu_matrix_mult(float *A, float *B, float *C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));

    // Initialize matrices with random values
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    double start = omp_get_wtime();
    cpu_matrix_mult(A, B, C, N);
    double time = omp_get_wtime() - start;

    printf("CPU Time: %.3f seconds\n", time);

    free(A);
    free(B);
    free(C);
    return 0;
}