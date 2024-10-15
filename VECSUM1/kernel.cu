#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <random> 

float sum_cpu(const std::vector<float>& vec) {
    auto start = std::chrono::high_resolution_clock::now();
    float sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "CPU: Summ = " << sum << ", Time = " << duration.count() << " sec\n";
    return sum;
}

__global__ void sum_reduction(float* input, float* output, int n) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp_sum = 0.0f;
    while (tid < n) {
        temp_sum += input[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp_sum;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }
    if (cacheIndex == 0) {
        output[blockIdx.x] = cache[0];
    }
}

float sum_gpu_launcher(const std::vector<float>& vec) {
    int N = vec.size();

    float* d_input = nullptr;
    float* d_output = nullptr;
    float h_output = 0.0f;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::vector<float> partial_sums(numBlocks, 0.0f);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    cudaMemcpy(d_input, vec.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    sum_reduction << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_sums.data(), d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    h_output = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0f);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "GPU: Summ = " << h_output << ", Time = " << duration.count() << " sec\n";
    cudaFree(d_input);
    cudaFree(d_output);

    return h_output;
}
int main() {
    int N = 1000000000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    std::vector<float> vec(N);
    for (int i = 0; i < N; ++i) {
        vec[i] = dis(gen);
    }
    float cpu_sum = sum_cpu(vec);
    float gpu_sum = sum_gpu_launcher(vec);
    float diff = std::abs(cpu_sum - gpu_sum);
    std::cout << "Difference between CPU and GPU results = " << diff << "\n";
    return 0;
}
