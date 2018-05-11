
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16
#define TILE_WIDTH30 30

namespace mxnet
{
namespace op
{
// __global__ void unroll(const float * input, float * output, const int B, const int M, const int C, const int H, const int W, const int K){
//    int tx = blockIdx.x * blockDim.x + threadIdx.x;
//    int tb = blockIdx.y * blockDim.y + threadIdx.y;
//
//    int c,s,h_out,w_out,unroll_w,unroll_h,w_base;
//
//    int H_out = H - K + 1;
//    int W_out = W - K + 1;
//    int UNROLLWIDTH = H_out * W_out;  // total
//
//    // if tb is valid
//
//    // i3:batch number  i2:channel number  i1:fiter row number i0:filter col number
//    //#define input4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//    // i2:batch number  i1:unroalled row number  i0:unrolled col number(filer #)
//    //#define output3d(i2, i1, i0) output[(i2) * ( UNROLLWIDTH * (C * K * K) ) + (i1) *(UNROLLWIDTH) + (i0)]
//
//    if ((tx < C * UNROLLWIDTH ) && (tb < B)) {
//      c = tx/UNROLLWIDTH;
//      s = tx%UNROLLWIDTH;
//
//      h_out = s/W_out;
//      w_out = s%W_out;
//      unroll_w = h_out * W_out + w_out;
//      w_base = c * K * K; // row base number
//
//      for(int p = 0; p < K; p++) {
//       for(int q = 0; q < K; q++) {
//         unroll_h = w_base + (p * K + q);
//         // output3d(tb,unroll_h,unroll_w) = input4d(tb, c, (h_out + p), (w_out + q));
//         output[(tb) * ( UNROLLWIDTH * (C * K * K) ) + (unroll_h) *(UNROLLWIDTH) + (unroll_w)] = input[(tb) * (C * H * W) + (c) * (H * W) + (h_out + p) * (W) + (w_out + q)];
//        }
//       }
//    }
//    //#undef output3d
//    //#undef input4d
// }


//
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//
//
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     const int UNROLLWIDTH = H_out * W_out;
//
// // An example use of these macros:
// // float a = y4d(0,0,0,0)
// // y4d(0,0,0,0) = a
// // i3:batch number  i2:channel number  i1:row number  i0:col row
// //#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + (i0)]
//
// //#define y3d(i2, i1, i0) y4d(i2, i1, ((i0)/K), ((i0)%K))
// // i2:batch number  i1:unroalled row number  i0:unrolled col number(filer #)
// //#define x3d(i2, i1, i0) x[(i2) * ( UNROLLWIDTH * (C * K * K) ) + (i1) *(UNROLLWIDTH) + (i0)]
// // i3:feature number  12:channel number  i1:row number  i0:col number
// //#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + (i0)]
// // i2:feature number  11:channel number  i0:element # in filter
// //#define k3d(i2, i1, i0) k4d(i2, i1, ((i0)/(K*K)), ((i0)%(K*K)))
//
//   int Row = blockIdx.y * blockDim.y + threadIdx.y ;
//   int Col = blockIdx.x * blockDim.x + threadIdx.x ;
//   int tb = blockIdx.z * blockDim.z + threadIdx.z ;
//
//   int numARows = M;
//   int numBColumns = UNROLLWIDTH;
//   int numAColumns = K * K * C;
//
//   if  ( (Row < numARows) && (Col < numBColumns) && (tb < B)){
//     float value = 0;
//
//     for( int j = 0 ; j < numAColumns ; j++){
//
//       //value = value + A[Row *numAColumns + k] *B[k*numBColumns +Col] ;
//       //value = value + (k4d(Row,j/(K*K),(j%(K*K)/K),(j%(K*K)%K))) * (x3d(tb,j,Col)) ;
//       value = value + (k[Row *numAColumns + j]) * (x[(tb) * ( UNROLLWIDTH * (C * K * K) ) + (j) *(UNROLLWIDTH) + (Col)]) ;
//
//     }
//       //C[Row*numCColumns +Col] = value ;
//       // y4d(tb, Row, (Col/K), (Col%K)) = value;
//       y[(tb) * (M * UNROLLWIDTH) + Row*UNROLLWIDTH + Col] = value;
//   }
//
//
// //#undef y4d
// //#undef y3d
// //#undef x3d
// //#undef k4d
// //#undef k3d
// }

//
// __global__ void shared_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//   __shared__ float TM[16][16];
//   __shared__ float TN[16][16];
//   const int UNROLLWIDTH = (H - K + 1) * (W - K + 1);
//
//   int Row = blockIdx.y * blockDim.y + threadIdx.y ;
//   int Col = blockIdx.x * blockDim.x + threadIdx.x ;
//   int tb = blockIdx.z * blockDim.z + threadIdx.z ;
//
//   int numARows = M;
//   int numBColumns = UNROLLWIDTH;
//   int numAColumns = K * K * C;
//
//
//   float PValue = 0;
//   if (tb < B) {
//     for( int i = 0; i< ceil(numAColumns/16.0); i++){
//         if((Row < numARows) && ((i*16 + threadIdx.x) < numAColumns ) ) {
//           TM[threadIdx.y][threadIdx.x] = k[Row * numAColumns+ i *16 +threadIdx.x];
//         }
//         else{
//           TM[threadIdx.y][threadIdx.x] = 0;
//         }
//
//         if(( (i*16+threadIdx.y) < numAColumns )  && (Col < numBColumns) ){
//           TN[threadIdx.y][threadIdx.x] = x[(tb) * ( UNROLLWIDTH * (C * K * K) ) + (i*16+threadIdx.y) * numBColumns + Col];
//         }
//         else{
//          TN[threadIdx.y][threadIdx.x] = 0 ;
//         }
//        __syncthreads();
//        for(int kk = 0; kk <16 ; kk++){
//           PValue = PValue + TM[threadIdx.y][kk] * TN[kk][threadIdx.x];
//        }
//        __syncthreads();
//        }
//     if((Row < numARows) && (Col<numBColumns)){
//      y[(tb) * (M * UNROLLWIDTH) + Row*numBColumns + Col] = PValue;
//     }
//   }
// }


__global__ void fuse_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
  #define UNROLLWIDTH 676
  #define numARows 16
  #define numBColumns 676
  #define numAColumns 150
  #define CHW 5400
  #define KK 25
  #define HW 900
  __shared__ float TM[16][16];
  __shared__ float TN[16][16];
  // const int UNROLLWIDTH = (H - K + 1) * (W - K + 1);

  int Row = blockIdx.y * blockDim.y + threadIdx.y ;
  int Col = blockIdx.x * blockDim.x + threadIdx.x ;
  int tb = blockIdx.z * blockDim.z + threadIdx.z ;

  // int numARows = M;
  // int numBColumns = UNROLLWIDTH;
  // int numAColumns = K * K * C;

  #define x4d(tb, i, Col, y) x[(tb) * (CHW) + \
                                  ((i*16+y)/(KK)) * (HW) + \
                                  ((((i*16+y)%(KK))/K)+(Col/(W-K+1)))*(W) + \
                                  (((i*16+y)%K)+(Col%(W-K+1)))]
  #define filter3d(Row,i,x) k[Row * numAColumns+ i *16 +x]
  #define y3d(tb,Row,Col)  y[(tb) * (M * UNROLLWIDTH) + Row*numBColumns + Col]

  float PValue = 0;
  if (tb < B) {
    for( int i = 0; i< ceil(numAColumns/16.0); i++)
    {
        if((Row < numARows) && ((i*16 + threadIdx.x) < numAColumns ) ) {
          TM[threadIdx.y][threadIdx.x] = filter3d(Row,i,threadIdx.x);
        }
        else{
          TM[threadIdx.y][threadIdx.x] = 0;
        }

        if(( (i*16+threadIdx.y) < numAColumns )  && (Col < numBColumns) ){
          // row::threadIdx.x   col::threadIdx.y
          TN[threadIdx.y][threadIdx.x] = x4d(tb,i,Col,threadIdx.y);
        }
        else{
         TN[threadIdx.y][threadIdx.x] = 0 ;
        }

       __syncthreads();
        // for(int kk = 0; kk <16 ; kk++){
        //    PValue = PValue + TM[threadIdx.y][kk] * TN[kk][threadIdx.x];
        // }
          PValue = PValue + TM[threadIdx.y][0] * TN[0][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][1] * TN[1][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][2] * TN[2][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][3] * TN[3][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][4] * TN[4][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][5] * TN[5][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][6] * TN[6][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][7] * TN[7][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][8] * TN[8][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][9] * TN[9][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][10] * TN[10][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][11] * TN[11][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][12] * TN[12][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][13] * TN[13][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][14] * TN[14][threadIdx.x];
          PValue = PValue + TM[threadIdx.y][15] * TN[15][threadIdx.x];

       __syncthreads();
       }
    if((Row < numARows) && (Col<numBColumns)){
     y3d(tb,Row,Col) = PValue;
    }
  }
  #undef x4d
  #undef y3d
  #undef filter3d
  #undef UNROLLWIDTH
  #undef numARows
  #undef numBColumns
  #undef numAColumns
  #undef CHW
  #undef KK
  #undef HW
}

__global__ void forward_conv_kernel(float *Y, const float *X,const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
{
  #define H_out 60
  #define W_out 60
  #define X_tile_width 20
  #define h_base  ((blockIdx.z/W_grid)*TILE_WIDTH)
  #define w_base  ((blockIdx.z%W_grid)*TILE_WIDTH)
  #define xtilearea 400
  #define sharedaccess (threadIdx.y * 5 + threadIdx.x)
  #define filteraccess  (blockIdx.y * 25 + threadIdx.y * 5 +threadIdx.x)
  #define Yaccess (blockIdx.x*21600 + blockIdx.y*3600 + (h_base+threadIdx.y) * 60 + w_base + threadIdx.x)
  #define CHW 4096
  #define HW 4096
  #define MHOWO 21600
  #define HOWO 3600
  #define CKK 25
  // const int H_out = H - K + 1;
  // const int W_out = W - K + 1;
  // int X_tile_width = TILE_WIDTH + K -1
  extern __shared__ float shmem[];
  float* X_shared = &shmem[0];
  float* W_shared = &shmem[xtilearea];
  int n = blockIdx.x;
  int m = blockIdx.y;
  int h0 = threadIdx.y;
  int w0 = threadIdx.x;
  // int h_base = (blockIdx.z/W_grid)*TILE_WIDTH;
  // int w_base = (blockIdx.z%W_grid)*TILE_WIDTH;
  int h = h_base + h0;
  int w = w_base + w0;
  float acc = 0.0f;
  int c = 0;
    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[filteraccess];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*CHW + c*HW+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
    }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }
      // if ((w+16) <(w_base + X_tile_width)) {
      //   if (i<H ) {
      //     X_shared[(i-h_base)*X_tile_width + w +16 - w_base] = X[n*C*H*W + c*H*W+ i*W + w+16];
      //   }
      //   else{
      //     X_shared[(i-h_base)*X_tile_width + w +16- w_base] = 0.0;
      //   }
      // }

    //   if (h < H && w < W) {
    //     X_shared[(h-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ h*W + w];
    //   }
    //   else{
    //     X_shared[(h-h_base)*X_tile_width + w - w_base] = 0.0;
    //   }
    //   if ((w+16) <(w_base + X_tile_width)) {
    //     if (h <H && (w+16<W)) {
    //       X_shared[(h-h_base)*X_tile_width + w +16 - w_base] = X[n*C*H*W + c*H*W+ h*W + w+16];
    //     }
    //     else{
    //       X_shared[(h-h_base)*X_tile_width + w +16- w_base] = 0.0;
    //     }
    // }
    // if ((h + 16) < (h_base + X_tile_width)) {
    //   if ((h+16) <H && w < W) {
    //     X_shared[(h+16-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ (h+16)*W + w];
    //   }
    //   else{
    //     X_shared[(h+16-h_base)*X_tile_width + w - w_base] = 0.0;
    //   }
    //   if ((w+16) <(w_base + X_tile_width)) {
    //     if ((h+16) <H && (w+16<W)) {
    //       X_shared[(h+16-h_base)*X_tile_width + w +16 - w_base] = X[n*C*H*W + c*H*W+ (h+16)*W + w+16];
    //     }
    //     else{
    //       X_shared[(h+16-h_base)*X_tile_width + w +16- w_base] = 0.0;
    //     }
    // }
  // }
    __syncthreads();
    // for(int p =0; p < K; p++) {
    //     for(int q = 0; q < K; q++) {
    //         acc += X_shared[(h0 + p) * X_tile_width + w0 + q] * W_shared[p * K + q];
    //     }
    // }
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];

    __syncthreads();
  if(h < H_out && w < W_out) {
     Y[Yaccess] = acc;

  }
    #undef W_out
    #undef H_out
    #undef X_tile_width
    #undef h_base
    #undef w_base
    #undef xtilearea
    #undef sharedaccess
    #undef filteraccess
    #undef Yaccess
    #undef CHW
    #undef HW
    #undef MHOWO
    #undef HOWO
    #undef CKK
}
__global__ void forward_conv30_kernel(float *Y, const float *X,const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
{
  #define H_out 26
  #define W_out 26
  #define X_tile_width 30
  #define h_base  ((blockIdx.z/W_grid)*TILE_WIDTH30)
  #define w_base  ((blockIdx.z%W_grid)*TILE_WIDTH30)
  #define sharedaccess (threadIdx.y * 5 + threadIdx.x)
  #define filteraccess  (blockIdx.y * 25 + threadIdx.y * 5 +threadIdx.x)
  #define Yaccess (blockIdx.x*10816 + blockIdx.y*676 + (h_base+threadIdx.y) * 26 + w_base + threadIdx.x)
  // const int H_out = H - K + 1;
  // const int W_out = W - K + 1;
  // int X_tile_width = TILE_WIDTH30 + K - 1;
  extern __shared__ float shmem[];
  float* X_shared = &shmem[0];
  float* W_shared = &shmem[X_tile_width * X_tile_width];
  int n = blockIdx.x;
  int m = blockIdx.y;
  int h0 = threadIdx.y;
  int w0 = threadIdx.x;
  // int h_base = (blockIdx.z/W_grid)*TILE_WIDTH30;
  // int w_base = (blockIdx.z%W_grid)*TILE_WIDTH30;
  int h = h_base + h0;
  int w = w_base + w0;
  float acc = 0.0f;
  // for(int c = 0; c < C; c++)
  // {
    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[m * C*K*K + 0 *K*K + h0 *K + w0];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH30) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH30) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*C*H*W + 0*H*W+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }

    }
    __syncthreads();
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];
    __syncthreads();


    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[m * C*K*K + 1 *K*K + h0 *K + w0];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH30) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH30) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*C*H*W + 1*H*W+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }

    }
    __syncthreads();
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];
    __syncthreads();


    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[m * C*K*K + 2 *K*K + h0 *K + w0];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH30) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH30) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*C*H*W + 2*H*W+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }

    }
    __syncthreads();
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];
    __syncthreads();

    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[m * C*K*K + 3 *K*K + h0 *K + w0];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH30) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH30) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*C*H*W + 3*H*W+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }

    }
    __syncthreads();
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];
    __syncthreads();

    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[m * C*K*K + 4 *K*K + h0 *K + w0];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH30) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH30) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*C*H*W + 4*H*W+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }

    }
    __syncthreads();
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];
    __syncthreads();

    if(h0 < K && w0 < K)
    {
      W_shared[sharedaccess] = k[m * C*K*K + 5 *K*K + h0 *K + w0];
    }
    __syncthreads();
    for(int i = h; i < h_base + X_tile_width; i+=TILE_WIDTH30) {
      for(int j = w; j < w_base + X_tile_width; j+=TILE_WIDTH30) {
        if(i < H && j < W)
          X_shared[(i-h_base)*X_tile_width + j - w_base] = X[n*C*H*W + 5*H*W+ i*W + j];
        else
          X_shared[(i-h_base)*X_tile_width + j - w_base] = 0.0;
      }
      // if (i<H) {
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = X[n*C*H*W + c*H*W+ i*W + w];
      // }
      // else{
      //   X_shared[(i-h_base)*X_tile_width + w - w_base] = 0.0;
      // }

    }
    __syncthreads();
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 0] * W_shared[0 * K + 0];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 1] * W_shared[0 * K + 1];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 2] * W_shared[0 * K + 2];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 3] * W_shared[0 * K + 3];
    acc += X_shared[(h0 + 0) * X_tile_width + w0 + 4] * W_shared[0 * K + 4];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 0] * W_shared[1 * K + 0];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 1] * W_shared[1 * K + 1];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 2] * W_shared[1 * K + 2];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 3] * W_shared[1 * K + 3];
    acc += X_shared[(h0 + 1) * X_tile_width + w0 + 4] * W_shared[1 * K + 4];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 0] * W_shared[2 * K + 0];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 1] * W_shared[2 * K + 1];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 2] * W_shared[2 * K + 2];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 3] * W_shared[2 * K + 3];
    acc += X_shared[(h0 + 2) * X_tile_width + w0 + 4] * W_shared[2 * K + 4];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 0] * W_shared[3 * K + 0];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 1] * W_shared[3 * K + 1];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 2] * W_shared[3 * K + 2];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 3] * W_shared[3 * K + 3];
    acc += X_shared[(h0 + 3) * X_tile_width + w0 + 4] * W_shared[3 * K + 4];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 0] * W_shared[4 * K + 0];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 1] * W_shared[4 * K + 1];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 2] * W_shared[4 * K + 2];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 3] * W_shared[4 * K + 3];
    acc += X_shared[(h0 + 4) * X_tile_width + w0 + 4] * W_shared[4 * K + 4];
    __syncthreads();
  // }
  //
  if(h < H_out && w < W_out) {
    Y[Yaccess] = acc;
  }
  #undef W_out
  #undef H_out
  #undef X_tile_width
  #undef h_base
  #undef w_base
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

    int NUM_THREADS = 16;
    int NUM_BATCH = 16;
    int NUM_BATCH_ = 1;
    // if (H == 30) {
    //   TILE_WIDTH = 30;
    // }
    //  if (H == 64) {
    //   TILE_WIDTH = 32;
    // }
    // size_t size = sizeof(float) * C * UNROLLWIDTH * K * K * B;
    // float * unrolled;
    // MSHADOW_CUDA_CALL(cudaMalloc((void **) &unrolled, size));
    // int H_out = H-K+1;
    // int W_out = W-K+1;
    int numthreads =  C * H_out * W_out;
    int numblocks = ceil(numthreads*1.0/NUM_THREADS);
    int numbatchs = ceil(B*1.0/NUM_BATCH);
    dim3 ugridDim(numblocks,numbatchs,1);
    dim3 ublockDim(NUM_THREADS,NUM_BATCH,1);
    // unroll<<<ugridDim,ublockDim>>>(x.dptr_,unrolled,B,M,C,H,W,K);
    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    // Set the kernel dimensions
    int mmH_grid = ceil(M*1.0/float(TILE_WIDTH));
    int mmW_grid = ceil(UNROLLWIDTH*1.0/float(TILE_WIDTH));
    int mmB_grid = ceil(B*1.0/NUM_BATCH_);
    int W_grid = ceil(W_out/(TILE_WIDTH*1.0));
    int H_grid = ceil(H_out/(TILE_WIDTH*1.0));
    int Z = H_grid * W_grid;

    int W_grid30 = ceil(W_out/(TILE_WIDTH30*1.0));
    int H_grid30 = ceil(H_out/(TILE_WIDTH30*1.0));
    int Z30 = H_grid30 * W_grid30;

    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, NUM_BATCH_);
    dim3 mmgridDim(mmW_grid,mmH_grid,mmB_grid);
    dim3 gridDim30(B,M,Z30);
    dim3 blockDim30(TILE_WIDTH30,TILE_WIDTH30,NUM_BATCH_);

    printf("B:%d\n", B);
    printf("M:%d\n", M);
    printf("C:%d\n", C);
    printf("H:%d\n", H);
    printf("W:%d\n", W);
    printf("K:%d\n", K);
    size_t s = sizeof(float) * ((TILE_WIDTH+K-1) * (TILE_WIDTH+K-1));
    size_t s30 = sizeof(float) * ((TILE_WIDTH30+K-1) * (TILE_WIDTH30+K-1));
    // int s = sizeof(float) * ((TILE_WIDTH + K-1)*(TILE_WIDTH + K-1) + K*K );
    // Call the kernel
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,unrolled,w.dptr_, B,M,C,H,W,K);
    // shared_forward_kernel<<<gridDim, blockDim>>>(y.dptr_,unrolled,w.dptr_, B,M,C,H,W,K);
    if (H == 30) {
      fuse_forward_kernel<<<mmgridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
      // forward_conv30_kernel<<<gridDim30, blockDim30,s30>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K,W_grid30);
    }
    else{
      forward_conv_kernel<<<gridDim, blockDim,s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K,W_grid);
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    // MSHADOW_CUDA_CALL(cudaFree(unrolled));
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
