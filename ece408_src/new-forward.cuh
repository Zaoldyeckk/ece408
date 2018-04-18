
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__global__ void unrolled(const float * input, float * output, const int B, const int M, const int C, const int H, const int W, const int K){
   int tx = blockIdx.x * blockDim.x + threadIdx.x;
   int tb = blockIdx.y * blockDim.y + threadIdx.y;

   int H_out = H - K + 1;
   int W_out = W - K + 1;
   int W_unroll = H_out * W_out;

   if (tx < C * W_unroll ) {
     /* code */
   }
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]



#undef y4d
#undef x4d
#undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <typename gpu>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    int NUM_THREADS = 128;
    int NUM_BATCH = 8;
    int TILE_WIDTH = 16;
    int size = sizeof(float) * C * (H-K+1) * (W- K + 1) * K * K * B;
    float * unrolled;
    cudaMalloc((void **), &unrolled, size );
    int H_out = H-K+1;
    int W_out = W-K+1;
    int numthreads =  C * H_out * W_out;
    int numblocks = ceil(numthreads*(1.0)/NUM_THREADS);
    int numbatchs = ceil(B*(1.0)/NUM_BATCH);
    dim3 ugridDim(numblocks,numbatchs,1);
    dim3 ublockDim(NUM_THREADS,NUM_BATCH,1);
    unrolled<<<ugridDim,ublockDim>>>(x.dptr_, unrolled,B,M,C,H,W,K);
    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    // Set the kernel dimensions
    int H_grid = ceil(H/16.0);
    int W_grid = ceil(W/16.0);


    int Z = H_grid * W_grid;
    dim3 gridDim(,M,Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    int s = sizeof(float) * ((TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );
    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,unrolled,w.dptr_, B,M,C,H,W,K);
    cudaFree(unrolled);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
