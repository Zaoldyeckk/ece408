
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

   int c,s,h_out,w_out,unroll_w,unroll_h,w_base;

   int H_out = H - K + 1;
   int W_out = W - K + 1;
   int UNROLLWIDTH = H_out * W_out;  // total

   // if tb is valid
   if (tb >= B){
     return ;
   }
   // i3:batch number  i2:channel number  i1:fiter row number i0:filter col number
   #define input4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
   // i2:batch number  i1:unroalled row number  i0:unrolled col number(filer #)
   #define output3d(i2, i1, i0) output[(i2) * ( UNROLLWIDTH * (C * K * K) ) + (i1) *(UNROLLWIDTH) + (i0)]

   if (tx < C * UNROLLWIDTH ) {
     c = tx/UNROLLWIDTH;
     s = tx%UNROLLWIDTH;

     h_out = s/W_out;
     w_out = s%W_out;
     unroll_w = h_out * W_out + w_out;
     w_base = c * K * K; // row base number
     for(int p = 0; p < K; p++) {
      for(int q = 0; q < K; q++) {
        unroll_h = w_base + (p * K + q);
        output3d(tb,unroll_h,unroll_w) = input4d(tb, c, (h_out + p), (w_out + q));
       }
      }
   }
   #undef output3d
   #undef input4d
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
    const int UNROLLWIDTH = H_out * W_out;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
// i3:batch number  i2:channel number  i1:row number  i0:col row
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

#define y3d(i2, i1, i0) y4d(i2, i1, (i0/K), (i0%K))
// i2:batch number  i1:unroalled row number  i0:unrolled col number(filer #)
#define x3d(i2, i1, i0) x[(i2) * ( UNROLLWIDTH * (C * K * K) ) + (i1) *(UNROLLWIDTH) + (i0)]
// i3:feature number  12:channel number  i1:row number  i0:col number
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
// i2:feature number  11:channel number  i0:element # in filter
#define k3d(i2, i1, i0) k4d(i2, i1, (i0/(K*K)), (i0%(K*K)))

  int Row = blockIdx.y * blockDim.y + threadIdx.y ;
  int Col = blockIdx.x * blockDim.x + threadIdx.x ;
  int tb = blockIdx.z * blockDim.z + threadIdx.z ;
  if (tb >= B){
    return ;
  }
  int numARows = M;
  int numBColumns = UNROLLWIDTH;
  int numAColumns = K * K * C;

  if  ( (Row < numARows) && (Col < numBColumns) ){
    float value = 0;
    for( int k = 0 ; k < numAColumns ; k++){


      //value = value + A[Row *numAColumns + k] *B[k*numBColumns +Col] ;
      value = value + k3d(tb,Row,k) * x3d(tb,k,Col) ;


    }
      //C[Row*numCColumns +Col] = value ;
      y3d(tb, Row, Col) = value;
  }


#undef y4d
#undef y3d
#undef x3d
#undef k4d
#undef k3d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
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
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int UNROLLWIDTH = H_out * W_out;

    int NUM_THREADS = 128;
    int NUM_BATCH = 8;
    int NUM_BATCH_ = 4;
    int TILE_WIDTH = 16;
    int size = sizeof(float) * C * UNROLLWIDTH * K * K * B;
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
    int H_grid = ceil(M*1.0/16.0);
    int W_grid = ceil(UNROLLWIDTH*1.0/16.0);
    int B_grid = ceil(B*1.0/NUM_BATCH_);
    dim3 gridDim(W_grid,H_grid,B_grid);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, NUM_BATCH_);
    // int s = sizeof(float) * ((TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );
    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,unrolled,w.dptr_, B,M,C,H,W,K);
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
