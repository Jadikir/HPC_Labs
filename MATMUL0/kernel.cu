#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
const int N = 2000;

// CUDA Kernel для перемножения матриц
__global__ void matrixMulKernel(int* C, const int* A, const int* B, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int result = 0;
        for (int k = 0; k < width; ++k) {
            result += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = result;
    }
}

// Функция умножения матриц на GPU
void matrixMulCUDA(int* C, const int* A, const int* B, int width) {
    int* d_A, * d_B, * d_C;
    size_t size = width * width * sizeof(int);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel <<<blocksPerGrid, threadsPerBlock >> > (d_C, d_A, d_B, width);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


void matrixMulCPU(int* C, const int* A, const int* B, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            C[row * width + col] = 0;
            for (int k = 0; k < width; ++k) {
                C[row * width + col] += A[row * width + k] * B[k * width + col];
            }
        }
    }
}


bool compareMatrices(const int* A, const int* B, int width) {
    for (int i = 0; i < width * width; ++i) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}

void measureTime(void(*func)(int*, const int*, const int*, int), int* C, const int* A, const int* B, int width, const char* description) {
    auto start = std::chrono::high_resolution_clock::now();
    func(C, A, B, width);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("%s took %f seconds\n", description, diff.count());
}

int main() {
    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C = new int[N * N];
    int* C_CPU = new int[N * N];
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }
    measureTime(matrixMulCUDA, C, A, B, N, "CUDA");
    measureTime(matrixMulCPU, C_CPU, A, B, N, "CPU");
    if (compareMatrices(C, C_CPU, N)) {
        printf("Results are correct!\n");
    }
    else {
        printf("Results are incorrect!\n");
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_CPU;
    return 0;
}