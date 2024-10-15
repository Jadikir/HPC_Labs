#include <iostream>
#include <omp.h> 
#include <chrono>
#include <cstdlib>


const int N = 100;

void matrixMulCPU(int* C, const int* A, const int* B, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            int result = 0;
            for (int k = 0; k < width; ++k) {
                result += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = result;
        }
    }
}


void matrixMulCPUParallel(int* C, const int* A, const int* B, int width) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            int result = 0;
            for (int k = 0; k < width; ++k) {
                result += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = result;
        }
    }
}

void initializeMatrix(int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = rand() % 10; 
    }
}

void printMatrix(const int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}
bool compareMatrices(const int* C1, const int* C2, int size) {
    return std::memcmp(C1, C2, size * size * sizeof(int)) == 0;
}
int main() {
    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C_seq = new int[N * N];
    int* C_par = new int[N * N];

    initializeMatrix(A, N * N);
    initializeMatrix(B, N * N);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(C_seq, A, B, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end - start; 
    std::cout << "Sequential Matrix Multiplication Time: " << duration_seq.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    matrixMulCPUParallel(C_par, A, B, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end - start;
    std::cout << "Parallel Matrix Multiplication Time: " << duration_par.count() << " seconds" << std::endl;

    if (compareMatrices(C_seq, C_par, N)) {
        std::cout << "Parallel and Sequential results match!" << std::endl;
    }
    else {
        std::cout << "Error: Parallel and Sequential results do not match!" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_par;

    return 0;
}
