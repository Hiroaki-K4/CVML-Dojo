#include <iostream>
#include <cuda_runtime.h>

__global__ void kernelOperation(float* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] = d_data[idx] * 2.0f;
    }
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float* h_data = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_data[i] = i * 0.5f;
    }

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Async copy and kernel execution
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    kernelOperation<<<(n + 255) / 256, 256, 0, stream>>>(d_data, n);
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Verify result
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << std::endl;
    }

    // Free memory
    free(h_data);
    cudaFree(d_data);
    cudaStreamDestroy(stream);

    return 0;
}
