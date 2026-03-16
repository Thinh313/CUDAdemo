#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1024   // matrix size

// =======================
// CPU Matrix Multiplication
// =======================
void cpuMatrixMul(float* A, float* B, float* C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;

            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// =======================
// CUDA Kernel
// Each thread computes one element
// =======================
__global__ void gpuMatrixMul(float* A, float* B, float* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    size_t size = N * N * sizeof(float);

    // Host matrices
    float* h_A, * h_B, * h_C_cpu, * h_C_gpu;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // =========================
    // CPU computation
    // =========================
    auto cpu_start = std::chrono::high_resolution_clock::now();

    cpuMatrixMul(h_A, h_B, h_C_cpu);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time =
        std::chrono::duration<double>(cpu_end - cpu_start).count();

    // =========================
    // GPU memory
    // =========================
    float* d_A, * d_B, * d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // =========================
    // CUDA Kernel configuration
    // =========================
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // =========================
    // GPU computation
    // =========================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gpuMatrixMul << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // =========================
    // Print results
    // =========================
    std::cout << "CPU time: " << cpu_time << " seconds\n";
    std::cout << "GPU time: " << gpu_time / 1000.0 << " seconds\n";

    std::cout << "Speedup: " << cpu_time / (gpu_time / 1000.0) << "x\n";

    // =========================
    // Cleanup
    // =========================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}