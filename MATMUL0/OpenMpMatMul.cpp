#include <iostream>
#include <omp.h>  // For OpenMP
#include <chrono>
#include <cstdlib>


const int N = 2048;  // Size of the matrix (N x N)

// Function to multiply matrices on CPU (sequentially)
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

// Function to multiply matrices on CPU (in parallel)
void matrixMulCPUParallel(int* C, const int* A, const int* B, int width) {
#pragma omp parallel for collapse(2)  // Parallelize outer loops
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

// Function to initialize matrices
void initializeMatrix(int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = rand() % 10;  // Fill matrices with random numbers from 0 to 9
    }
}

// Function for matrix addition
void add(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] + B[i * size + j];
        }
    }
}

// Function for matrix subtraction
void subtract(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] - B[i * size + j];
        }
    }
}

// Strassen's Algorithm (basic version)
void strassen(int* C, const int* A, const int* B, int width) {
    if (width == 1) {
        C[0] = A[0] * B[0];
        return;
    }

    int newSize = width / 2;
    int* A11 = new int[newSize * newSize];
    int* A12 = new int[newSize * newSize];
    int* A21 = new int[newSize * newSize];
    int* A22 = new int[newSize * newSize];
    int* B11 = new int[newSize * newSize];
    int* B12 = new int[newSize * newSize];
    int* B21 = new int[newSize * newSize];
    int* B22 = new int[newSize * newSize];
    int* M1 = new int[newSize * newSize];
    int* M2 = new int[newSize * newSize];
    int* M3 = new int[newSize * newSize];
    int* M4 = new int[newSize * newSize];
    int* M5 = new int[newSize * newSize];
    int* M6 = new int[newSize * newSize];
    int* M7 = new int[newSize * newSize];

    // Divide A and B into 4 sub-matrices each
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i * newSize + j] = A[i * width + j];
            A12[i * newSize + j] = A[i * width + (j + newSize)];
            A21[i * newSize + j] = A[(i + newSize) * width + j];
            A22[i * newSize + j] = A[(i + newSize) * width + (j + newSize)];

            B11[i * newSize + j] = B[i * width + j];
            B12[i * newSize + j] = B[i * width + (j + newSize)];
            B21[i * newSize + j] = B[(i + newSize) * width + j];
            B22[i * newSize + j] = B[(i + newSize) * width + (j + newSize)];
        }
    }

    // Compute M1 to M7
    int* temp1 = new int[newSize * newSize];
    int* temp2 = new int[newSize * newSize];

    // M1 = (A11 + A22)(B11 + B22)
    add(A11, A22, temp1, newSize);
    add(B11, B22, temp2, newSize);
    strassen(M1, temp1, temp2, newSize);

    // M2 = (A21 + A22)B11
    add(A21, A22, temp1, newSize);
    strassen(M2, temp1, B11, newSize);

    // M3 = A11(B12 - B22)
    subtract(B12, B22, temp2, newSize);
    strassen(M3, A11, temp2, newSize);

    // M4 = A22(B21 - B11)
    subtract(B21, B11, temp2, newSize);
    strassen(M4, A22, temp2, newSize);

    // M5 = (A11 + A12)B22
    add(A11, A12, temp1, newSize);
    strassen(M5, temp1, B22, newSize);

    // M6 = (A21 - A11)(B11 + B12)
    subtract(A21, A11, temp1, newSize);
    add(B11, B12, temp2, newSize);
    strassen(M6, temp1, temp2, newSize);

    // M7 = (A12 - A22)(B21 + B22)
    subtract(A12, A22, temp1, newSize);
    add(B21, B22, temp2, newSize);
    strassen(M7, temp1, temp2, newSize);

    // Combine results into C
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i * width + j] = M1[i * newSize + j] + M4[i * newSize + j] - M5[i * newSize + j] + M7[i * newSize + j];
            C[i * width + (j + newSize)] = M3[i * newSize + j] + M5[i * newSize + j];
            C[(i + newSize) * width + j] = M2[i * newSize + j] + M4[i * newSize + j];
            C[(i + newSize) * width + (j + newSize)] = M1[i * newSize + j] - M2[i * newSize + j] + M3[i * newSize + j] + M6[i * newSize + j];
        }
    }

    // Free dynamically allocated memory
    delete[] A11;
    delete[] A12;
    delete[] A21;
    delete[] A22;
    delete[] B11;
    delete[] B12;
    delete[] B21;
    delete[] B22;
    delete[] M1;
    delete[] M2;
    delete[] M3;
    delete[] M4;
    delete[] M5;
    delete[] M6;
    delete[] M7;
    delete[] temp1;
    delete[] temp2;
}

// Function to multiply matrices on CPU (Strassen's algorithm in parallel)
void strassenParallel(int* C, const int* A, const int* B, int width) {
    if (width == 1) {
        C[0] = A[0] * B[0];
        return;
    }

    int newSize = width / 2;
    int* A11 = new int[newSize * newSize];
    int* A12 = new int[newSize * newSize];
    int* A21 = new int[newSize * newSize];
    int* A22 = new int[newSize * newSize];
    int* B11 = new int[newSize * newSize];
    int* B12 = new int[newSize * newSize];
    int* B21 = new int[newSize * newSize];
    int* B22 = new int[newSize * newSize];
    int* M1 = new int[newSize * newSize];
    int* M2 = new int[newSize * newSize];
    int* M3 = new int[newSize * newSize];
    int* M4 = new int[newSize * newSize];
    int* M5 = new int[newSize * newSize];
    int* M6 = new int[newSize * newSize];
    int* M7 = new int[newSize * newSize];

    // Divide A and B into 4 sub-matrices each
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i * newSize + j] = A[i * width + j];
            A12[i * newSize + j] = A[i * width + (j + newSize)];
            A21[i * newSize + j] = A[(i + newSize) * width + j];
            A22[i * newSize + j] = A[(i + newSize) * width + (j + newSize)];

            B11[i * newSize + j] = B[i * width + j];
            B12[i * newSize + j] = B[i * width + (j + newSize)];
            B21[i * newSize + j] = B[(i + newSize) * width + j];
            B22[i * newSize + j] = B[(i + newSize) * width + (j + newSize)];
        }
    }

    // Compute M1 to M7 in parallel
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = newSize / num_threads;

        if (tid == 0) {  // M1 calculation
            int* temp1 = new int[newSize * newSize];
            int* temp2 = new int[newSize * newSize];
            add(A11, A22, temp1, newSize);
            add(B11, B22, temp2, newSize);
            strassen(M1, temp1, temp2, newSize);
            delete[] temp1;
            delete[] temp2;
        }
        else if (tid == 1) {  // M2 calculation
            int* temp1 = new int[newSize * newSize];
            add(A21, A22, temp1, newSize);
            strassen(M2, temp1, B11, newSize);
            delete[] temp1;
        }
        else if (tid == 2) {  // M3 calculation
            int* temp2 = new int[newSize * newSize];
            int* temp1 = new int[newSize * newSize];
            subtract(B12, B22, temp2, newSize);
            strassen(M3, A11, temp2, newSize);
            delete[] temp2;
            delete[] temp1;
        }
        else if (tid == 3) {  // M4 calculation
            int* temp2 = new int[newSize * newSize];
            int* temp1 = new int[newSize * newSize];
            subtract(B21, B11, temp2, newSize);
            strassen(M4, A22, temp2, newSize);
            delete[] temp2;
            delete[] temp1;
        }
        else if (tid == 4) {  // M5 calculation
            int* temp1 = new int[newSize * newSize];
            add(A11, A12, temp1, newSize);
            strassen(M5, temp1, B22, newSize);
            delete[] temp1;
        }
        else if (tid == 5) {  // M6 calculation
            int* temp2 = new int[newSize * newSize];
            int* temp1 = new int[newSize * newSize];
            subtract(A21, A11, temp1, newSize);
            add(B11, B12, temp2, newSize);
            strassen(M6, temp1, temp2, newSize);
            delete[] temp2;
            delete[] temp1;
        }
        else if (tid == 6) {  // M7 calculation
            int* temp1 = new int[newSize * newSize];
            int* temp2 = new int[newSize * newSize];
            subtract(A12, A22, temp1, newSize);
            add(B21, B22, temp2, newSize);
            strassen(M7, temp1, temp2, newSize);
            delete[] temp2;
            delete[] temp1;
        }
    }

    // Combine results into C
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i * width + j] = M1[i * newSize + j] + M4[i * newSize + j] - M5[i * newSize + j] + M7[i * newSize + j];
            C[i * width + (j + newSize)] = M3[i * newSize + j] + M5[i * newSize + j];
            C[(i + newSize) * width + j] = M2[i * newSize + j] + M4[i * newSize + j];
            C[(i + newSize) * width + (j + newSize)] = M1[i * newSize + j] - M2[i * newSize + j] + M3[i * newSize + j] + M6[i * newSize + j];
        }
    }

    // Free dynamically allocated memory
    delete[] A11;
    delete[] A12;
    delete[] A21;
    delete[] A22;
    delete[] B11;
    delete[] B12;
    delete[] B21;
    delete[] B22;
    delete[] M1;
    delete[] M2;
    delete[] M3;
    delete[] M4;
    delete[] M5;
    delete[] M6;
    delete[] M7;
}

// Function to print the matrix
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
    // Allocate memory for matrices
    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C_seq = new int[N * N]; // Sequential multiplication result
    int* C_par = new int[N * N]; // Parallel multiplication result
    int* C_strassen = new int[N * N]; // Strassen's result
    int* C_strassen_par = new int[N * N]; // Strassen's parallel result

    // Initialize matrices
    initializeMatrix(A, N * N);
    initializeMatrix(B, N * N);

    // Sequential matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(C_seq, A, B, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    // Parallel matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    matrixMulCPUParallel(C_par, A, B, N);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    // Strassen's algorithm (sequential)
    start = std::chrono::high_resolution_clock::now();
    strassen(C_strassen, A, B, N);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Strassen's Algorithm Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    // Strassen's algorithm (parallel)
    start = std::chrono::high_resolution_clock::now();
    strassenParallel(C_strassen_par, A, B, N);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Strassen's Algorithm (Parallel) Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    // Verification of results using conditional statements
    if (compareMatrices(C_seq, C_par, N)) {
        std::cout << "Parallel and Sequential results match!" << std::endl;
    }
    else {
        std::cout << "Error: Parallel and Sequential results do not match!" << std::endl;
    }

    if (compareMatrices(C_seq, C_strassen, N)) {
        std::cout << "Strassen's result matches Sequential result!" << std::endl;
    }
    else {
        std::cout << "Error: Strassen's result does not match Sequential result!" << std::endl;
    }

    if (compareMatrices(C_seq, C_strassen_par, N)) {
        std::cout << "Strassen's Parallel result matches Sequential result!" << std::endl;
    }
    else {
        std::cout << "Error: Strassen's Parallel result does not match Sequential result!" << std::endl;
    }

    // Free dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_par;
    delete[] C_strassen;
    delete[] C_strassen_par;

    return 0;
}
