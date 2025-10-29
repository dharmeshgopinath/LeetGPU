#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

int i = blockIdx.x * blockDim.x + threadIdx.x;

if(i<N){
    C[i] = A[i] + B[i];
}
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    
    //Mem Alloacation in GPU -> VRAM
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc((void**) &d_A, N*sizeof(float));
    cudaMalloc((void**) &d_B, N*sizeof(float));
    cudaMalloc((void**) &d_C, N*sizeof(float));

    
    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize(); //used to wait for GPU to finish all processes

    // Free memory in device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
