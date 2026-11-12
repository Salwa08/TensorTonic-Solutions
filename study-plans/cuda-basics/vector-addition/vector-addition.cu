#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Store N element-wise results in C; this kernel returns no value.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < N) {
        C[workIndex] = A[workIndex] + B[workIndex];
}
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}