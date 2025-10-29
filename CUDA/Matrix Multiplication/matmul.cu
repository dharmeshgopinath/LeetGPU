#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    //Identifying the i,j index of the element in the output matrix C
    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if(row < M && col < K){ //out of bounds check; (M x K) is the dimensions of C Matrix 

        float value = 0.0f; //  0 float

        for(int k = 0; k < N; k++) //Here N is the shared dimension between A and B; remember it as the inner dimension from that one yt vid
        {
            //This is teh 
            value += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = value; //Storing result in output matrix C
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    
    float* d_A;
    float* d_B;
    float* d_C;

    //size calc for each matrix
    size_t sizeA = M * N *sizeof(float);
    size_t sizeB = N * K *sizeof(float);
    size_t sizeC = M * K *sizeof(float);

    //Mem Allocation in device
    cudaMalloc((void**) &d_A, sizeA);
    cudaMalloc((void**) &d_B, sizeB);
    cudaMalloc((void**) &d_C, sizeC);
    
    //Data transfer to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    
    
    //grid dim annd block dim calc    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free memory in device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
