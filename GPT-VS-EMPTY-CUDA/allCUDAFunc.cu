#include "cudaInclude.cuh"
#include "init_cuda.h"
#include "stdInte.cuh"

#pragma region DeviceMemoryPointer
void *d_cuda_defcan_vars_ptr;
void *d_image1_ptr, *d_image2_ptr;
void *d_g_can1_ptr, *d_g_ang1_ptr, *d_g_nor1_ptr, *d_gpt_ptr;
#pragma endregion DeviceMemoryPointer

int iDivUp(int hostPtr, int b) {
  return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b);
};
dim3 numBlock;
dim3 numThread;

void setGPUSize(int blockX, int blockY, int threadX, int threadY) {
  numBlock.x = iDivUp(blockX, threadX);
  numBlock.y = iDivUp(blockY, threadY);
  numThread.x = threadX;
  numThread.y = threadY;
}

template <typename T> __device__ void customAdd(T *sdata, T *g_odata) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  // do reduction in shared mem
  if (tid < 512) {
    sdata[tid] += sdata[tid + 512];
  }
  __syncthreads();
  if (tid < 256) {
    sdata[tid] += sdata[tid + 256];
  }
  __syncthreads();
  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
  }
  __syncthreads();
  if (tid < 16) {
    sdata[tid] += sdata[tid + 16];
  }
  __syncthreads();
  if (tid < 8) {
    sdata[tid] += sdata[tid + 8];
  }
  __syncthreads();
  if (tid < 4) {
    sdata[tid] += sdata[tid + 4];
  }
  __syncthreads();
  if (tid < 2) {
    sdata[tid] += sdata[tid + 2];
  }
  __syncthreads();
  if (tid < 1) {
    sdata[tid] += sdata[tid + 1];
  }
  __syncthreads();
  // write result for this block to global mem
  if (tid == 0) {
    atomicAdd(g_odata, sdata[tid]);
  }
}

#pragma region ProcImage

__global__ void cuda_defcan1() {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((y >= ROW) || (x >= COL)) {
    return;
  }

  /* definite canonicalization */
  int margine = CANMARGIN / 2;
  int condition = ((x >= margine && y >= margine) && (x < COL - margine) &&
                   (y < ROW - margine) && d_image1[y][x] != WHITE);

  float this_pixel = condition * (float)d_image1[y][x];
  __shared__ float sdata[3][TPB_X_TPB];
  sdata[0][tid] = this_pixel;
  sdata[1][tid] = this_pixel * this_pixel;
  sdata[2][tid] = condition;

  __syncthreads();

  customAdd(sdata[0], d_cuda_defcan_vars);
  customAdd(sdata[1], d_cuda_defcan_vars + 1);
  customAdd(sdata[2], d_cuda_defcan_vars + 2);
}
__global__ void cuda_defcan2() {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((y >= ROW) || (x >= COL)) {
    return;
  }

  /*
          s_vars[0]:  mean
          s_vars[1]:  norm
  */
  __shared__ float s_vars[2];
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float npo = d_cuda_defcan_vars[2];
    float mean = d_cuda_defcan_vars[0] / (float)npo;
    float norm = d_cuda_defcan_vars[1] - (float)npo * mean * mean;
    if (norm == 0.0)
      norm = 1.0;
    s_vars[0] = mean;
    s_vars[1] = norm;
  }
  __syncthreads();

  int condition = ((x < COL - CANMARGIN) && (y < ROW - CANMARGIN) &&
                   d_image1[y][x] != WHITE);

  float ratio = 1.0 / sqrt(s_vars[1]);
  d_g_can1[y][x] = condition * ratio * ((float)d_image1[y][x] - s_vars[0]);
}

__global__ void cuda_roberts8() {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((y >= ROW) || (x >= COL)) {
    return;
  }

  /* extraction of gradient information by Roberts operator */
  /* with 8-directional codes and strength */
  float delta_RD, delta_LD;
  float angle;

  /* angle & norm of gradient vector calculated
   by Roberts operator */

  if (y >= ROW - 1 || x >= COL - 1) {
    d_g_ang1[y][x] = -1;
    d_g_nor1[y][x] = 0.0;
    return;
  }

  delta_RD = d_image1[y][x + 1] - d_image1[y + 1][x];
  delta_LD = d_image1[y][x] - d_image1[y + 1][x + 1];
  d_g_nor1[y][x] = sqrt(delta_RD * delta_RD + delta_LD * delta_LD);

  if (d_g_nor1[y][x] == 0.0 ||
      delta_RD * delta_RD + delta_LD * delta_LD < NoDIRECTION * NoDIRECTION) {
    d_g_ang1[y][x] = -1;
    return;
  }
  if (abs(delta_RD) == 0.0) {
    if (delta_LD > 0)
      d_g_ang1[y][x] = 3;
    else if (delta_LD < 0)
      d_g_ang1[y][x] = 7;
    else
      d_g_ang1[y][x] = -1;
    return;
  }
  angle = atan2(delta_LD, delta_RD);
  if (angle * 8.0 > 7.0 * PI) {
    d_g_ang1[y][x] = 5;
    return;
  }
  if (angle * 8.0 > 5.0 * PI) {
    d_g_ang1[y][x] = 4;
    return;
  }
  if (angle * 8.0 > 3.0 * PI) {
    d_g_ang1[y][x] = 3;
    return;
  }
  if (angle * 8.0 > 1.0 * PI) {
    d_g_ang1[y][x] = 2;
    return;
  }
  if (angle * 8.0 > -1.0 * PI) {
    d_g_ang1[y][x] = 1;
    return;
  }
  if (angle * 8.0 > -3.0 * PI) {
    d_g_ang1[y][x] = 0;
    return;
  }
  if (angle * 8.0 > -5.0 * PI) {
    d_g_ang1[y][x] = 7;
    return;
  }
  if (angle * 8.0 > -7.0 * PI) {
    d_g_ang1[y][x] = 6;
    return;
  }
  d_g_ang1[y][x] = 5;
}

void procImageInitial() {
  gpuErrchk(cudaGetSymbolAddress(&d_image1_ptr, d_image1));
  gpuErrchk(cudaGetSymbolAddress(&d_g_can1_ptr, d_g_can1));
  gpuErrchk(cudaGetSymbolAddress(&d_g_nor1_ptr, d_g_nor1));
  gpuErrchk(cudaGetSymbolAddress(&d_g_ang1_ptr, d_g_ang1));
  gpuErrchk(cudaGetSymbolAddress(&d_cuda_defcan_vars_ptr, d_cuda_defcan_vars));
  gpuStop()
}

void cuda_procImg(float *g_can, int *g_ang, float *g_nor, unsigned char *image1,
                  int copy) {

  if (copy == 1)
    cudaMemcpy(d_image1_ptr, image1, ROW * COL * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

  setGPUSize(COL, ROW, TPB, TPB);
  cudaMemset(d_cuda_defcan_vars_ptr, 0, 3 * sizeof(float));
  cuda_defcan1<<<numBlock, numThread>>>();
  cuda_defcan2<<<numBlock, numThread>>>();
  cuda_roberts8<<<numBlock, numThread>>>();
  cudaMemcpy(g_can, d_g_can1_ptr, ROW * COL * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(g_ang, d_g_ang1_ptr, ROW * COL * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(g_nor, d_g_nor1_ptr, ROW * COL * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuStop()
}
#pragma endregion ProcImage

#pragma region Bilinear

void bilinearInitial() {
  gpuErrchk(cudaGetSymbolAddress(&d_gpt_ptr, d_gpt));
  gpuErrchk(cudaGetSymbolAddress(&d_image2_ptr, d_image2));
  gpuStop();
}

__global__ void cuda_calc_bilinear_normal_inverse_projection(int x_size1,
                                                             int y_size1,
                                                             int x_size2,
                                                             int y_size2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((y >= y_size1) || (x >= x_size1)) {
    return;
  }
  int cx, cy, cx2, cy2;
  if (y_size1 == ROW) {
    cx = CX, cy = CY;
    cx2 = CX2, cy2 = CY2;
  } else {
    cx = CX2, cy = CY2;
    cx2 = CX, cy2 = CY;
  }

  float inVect[3], outVect[3];
  float x_new, y_new, x_frac, y_frac;
  float gray_new;
  int m, n;

  inVect[2] = 1.0;
  inVect[1] = y - cy;
  inVect[0] = x - cx;

  int i, j;
  float sum;
  for (i = 0; i < 3; ++i) {
    sum = 0.0;
    for (j = 0; j < 3; ++j) {
      sum += d_gpt[i][j] * inVect[j];
    }
    outVect[i] = sum;
  }

  x_new = outVect[0] / outVect[2] + cx2;
  y_new = outVect[1] / outVect[2] + cy2;
  m = (int)floor(x_new);
  n = (int)floor(y_new);
  x_frac = x_new - m;
  y_frac = y_new - n;

  if (m >= 0 && m + 1 < x_size2 && n >= 0 && n + 1 < y_size2) {
    gray_new = (1.0 - y_frac) * ((1.0 - x_frac) * d_image2[n][m] +
                                 x_frac * d_image2[n][m + 1]) +
               y_frac * ((1.0 - x_frac) * d_image2[n + 1][m] +
                         x_frac * d_image2[n + 1][m + 1]);
    d_image1[y][x] = (unsigned char)gray_new;
  } else {
#ifdef BACKGBLACK
    d_image1[y][x] = BLACK;
#else
    d_image1[y][x] = WHITE;
#endif
  }
}

void cuda_bilinear_normal_inverse_projection(float gpt[3][3], int x_size1,
                                             int y_size1, int x_size2,
                                             int y_size2, unsigned char *image1,
                                             unsigned char *image2,
                                             int initial) {
  /* inverse projection transformation of the image by bilinear interpolation */

  if (initial == 1)
    cudaMemcpy(d_image2_ptr, image1, sizeof(unsigned char) * ROW2 * COL2,
               cudaMemcpyHostToDevice);
  gpuStop()

      cudaMemcpy(d_gpt_ptr, gpt, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  setGPUSize(COL, ROW, TPB, TPB);
  cuda_calc_bilinear_normal_inverse_projection<<<numBlock, numThread>>>(
      x_size1, y_size1, x_size2, y_size2);
}

#pragma endregion Bilinear

#pragma region SHoGPat

void *d_inteAng_ptr;
void *d_sHoG_ptr;

__device__ int d_dnnL[] = DNNL;
__device__ float d_dnn[1];
void *d_dnn_ptr;
__device__ int d_count[1];
void *d_count_ptr;

void sHoGpatInitial(int *inteAng) {
  gpuErrchk(cudaGetSymbolAddress(&d_inteAng_ptr, d_inteAng));
  gpuErrchk(cudaGetSymbolAddress(&d_sHoG_ptr, d_sHoG));
  gpuErrchk(cudaGetSymbolAddress(&d_dnn_ptr, d_dnn));
  gpuErrchk(cudaGetSymbolAddress(&d_count_ptr, d_count));
  cudaMemcpy(d_inteAng_ptr, inteAng, ROWINTE * COLINTE * 64 * sizeof(int),
             cudaMemcpyHostToDevice);
  gpuStop()
}

__global__ void cuda_sHoGpatInte(int nDnnL) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= COL - 4 || y >= ROW - 4) {
    return;
  }
  int maxWinP = MAXWINDOWSIZE + 1;
  int x1 = x + 2;
  int y1 = y + 2;
  int ang1 = 0;
  int count = 0;
  float dnn = 0;
  if ((ang1 = d_sHoG[y][x]) != -1) {
    for (int wN = 0; wN < nDnnL; wN++) {
      int pPos = maxWinP + d_dnnL[wN];
      int mPos = MAXWINDOWSIZE - d_dnnL[wN];
      float secInte = d_inteAng[y1 + pPos][x1 + pPos][ang1] -
                      d_inteAng[(y1 + pPos)][x1 + mPos][ang1] -
                      d_inteAng[(y1 + mPos)][x1 + pPos][ang1] +
                      d_inteAng[(y1 + mPos)][x1 + mPos][ang1];
      if (secInte > 0) {
        count = 1;
        dnn = d_dnnL[wN];
        break;
      }
    }
  }

  __shared__ float sdataD[TPB_X_TPB];
  __shared__ int sdataI[TPB_X_TPB];
  sdataD[tid] = dnn;
  sdataI[tid] = count;

  __syncthreads();

  customAdd(sdataD, d_dnn);
  customAdd(sdataI, d_count);
}

float sHoGpatInteGPU(int *sHoG1) {
  cudaMemset(d_count_ptr, 0, sizeof(int));
  cudaMemset(d_dnn_ptr, 0, sizeof(float));
  cudaMemcpy(d_sHoG_ptr, sHoG1, (ROW - 4) * (COL - 4) * sizeof(int),
             cudaMemcpyHostToDevice);

  int dnnL[] = DNNL;
  int nDnnL;
  int count = 0;
  float dnn = 0;

  for (int wN = 0; wN < NDNNL; ++wN) {
    if (dnnL[wN] >= MAXWINDOWSIZE) {
      nDnnL = wN + 1;
      break;
    }
  }

  setGPUSize(COL, ROW, TPB, TPB);
  cuda_sHoGpatInte<<<numBlock, numThread>>>(nDnnL);
  cudaMemcpy(&count, d_count_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&dnn, d_dnn_ptr, sizeof(float), cudaMemcpyDeviceToHost);
  gpuStop() float ddnn;
  if (count == 0)
    ddnn = MAXWINDOWSIZE;
  else
    ddnn = (float)dnn / count;
  return ddnn;
}

#pragma endregion SHoGPat
