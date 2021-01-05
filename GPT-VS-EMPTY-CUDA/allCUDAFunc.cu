#include "cudaInclude.cuh"
#include "init_cuda.h"
#include "stdInte.cuh"


#pragma region CommonDeviceMemory
__device__ unsigned char d_image1[ROW][COL];
__device__ unsigned char d_image2[ROW2][COL2];
__device__ double d_g_can1[ROW][COL], d_g_nor1[ROW][COL], d_gpt[3][3];
__device__ int d_g_ang1[ROW][COL], d_sHoG[ROW - 4][COL - 4];
__device__ int d_inteAng[64][ROWINTE][COLINTE];
__device__ double d_inteCanDir[64][ROWINTE][COLINTE];
__device__ double d_inteDx2Dir[64][ROWINTE][COLINTE];
__device__ double d_inteDy2Dir[64][ROWINTE][COLINTE];

void *d_image1_ptr,*d_image2_ptr;
void *d_g_can1_ptr, *d_g_ang1_ptr, *d_g_nor1_ptr,*d_gpt_ptr;
void *d_inteAng_ptr, *d_inteCanDir_ptr, *d_inteDx2Dir_ptr, *d_inteDy2Dir_ptr;
void *d_sHoG_ptr;
#pragma endregion CommonDeviceMemory

int iDivUp(int hostPtr, int b) { return ((hostPtr + b -1) / b); };
dim3 numBlock;
dim3 numThread;


void setGPUSize(int blockX, int blockY, int threadX, int threadY)
{
	numBlock.x = iDivUp(blockX, threadX);
	numBlock.y = iDivUp(blockY, threadY);
	numThread.x = threadX;
	numThread.y = threadY;
}

template<typename T>
__device__ void warpReduce(volatile T* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template<typename T>
__device__ void customAdd(T* sdata, T* g_odata) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * blockDim.x + tx;
	// do reduction in shared mem
	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32) warpReduce(sdata, tid); __syncthreads();
	//if (tid < 32) { sdata[tid] += sdata[tid + 32]; }__syncthreads();
	//if (tid < 16) { sdata[tid] += sdata[tid + 16]; }__syncthreads();
	//if (tid < 8) { sdata[tid] += sdata[tid + 8]; }__syncthreads();
	//if (tid < 4) { sdata[tid] += sdata[tid + 4]; }__syncthreads();
	//if (tid < 2) { sdata[tid] += sdata[tid + 2]; }__syncthreads();
	//if (tid < 1) { sdata[tid] += sdata[tid + 1]; }__syncthreads();
	// write result for this block to global mem
	if (tid == 0) { atomicAdd(g_odata, sdata[tid]); }

}

template<typename T>
__global__ void cuda_sumMatrix(T* martrix, T* out, int y_size, int x_size, int shift)
{
	int tid = threadIdx.x;
	int location = blockIdx.x*blockDim.x + threadIdx.x;
	int totalSize = x_size * y_size;

	__shared__ T sdata[TPB_X_TPB];
	if (location >= totalSize) {
		sdata[tid] = 0;
		return;
	}

	sdata[tid] = martrix[location + shift * totalSize];
	__syncthreads();
	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32) warpReduce(sdata, tid); __syncthreads();
	//if (tid < 32) { sdata[tid] += sdata[tid + 32]; }__syncthreads();
	//if (tid < 16) { sdata[tid] += sdata[tid + 16]; }__syncthreads();
	//if (tid < 8) { sdata[tid] += sdata[tid + 8]; }__syncthreads();
	//if (tid < 4) { sdata[tid] += sdata[tid + 4]; }__syncthreads();
	//if (tid < 2) { sdata[tid] += sdata[tid + 2]; }__syncthreads();
	//if (tid < 1) { sdata[tid] += sdata[tid + 1]; }__syncthreads();
	if (tid == 0) { atomicAdd(&out[shift], sdata[tid]); }
}

#pragma region ProcImage


__device__ double d_cuda_defcan_vars[4];
__device__ double d_cuda_defcan_vars_array[3 * ROW * COL];
double *d_cuda_defcan_vars_ptr;
double *d_cuda_defcan_vars_array_ptr;

__global__ void cuda_defcan1() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((y >= ROW) || (x >= COL)) {
		return;
	}
	/* definite canonicalization */
	int condition = (d_image1[y][x] != WHITE);

	double this_pixel = condition * (double)d_image1[y][x];

	d_cuda_defcan_vars_array[y*COL + x] = this_pixel;
	d_cuda_defcan_vars_array[y*COL + x + ROW * COL] = this_pixel * this_pixel;
	d_cuda_defcan_vars_array[y*COL + x + 2 * ROW * COL] = condition;
}
__global__ void cuda_defcan2() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((y >= ROW) || (x >= COL)) {
		return;
	}

	/*
		s_vars[0]:  mean
		s_vars[1]:  norm
	*/
	__shared__ double s_vars[2];
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		double npo = d_cuda_defcan_vars[2];
		double mean = d_cuda_defcan_vars[0] / (double)npo;
		double norm = d_cuda_defcan_vars[1] - (double)npo * mean * mean;
		if (norm == 0.0) norm = 1.0;
		s_vars[0] = mean;
		s_vars[1] = norm;
	}
	__syncthreads();

	int condition = ((x < COL - CANMARGIN) && (y < ROW - CANMARGIN) &&
		d_image1[y][x] != WHITE);

	double ratio = 1.0 / sqrt(s_vars[1]);
	d_g_can1[y][x] = condition * ratio * ((double)d_image1[y][x] - s_vars[0]);
}

__global__ void cuda_roberts8() {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((y >= ROW) || (x >= COL)) {
		return;
	}

	/* extraction of gradient information by Roberts operator */
	/* with 8-directional codes and strength */
	double delta_RD, delta_LD;
	double angle;

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

	if (d_g_nor1[y][x] == 0.0 || delta_RD * delta_RD + delta_LD * delta_LD < NoDIRECTION * NoDIRECTION) {
		d_g_ang1[y][x] = -1;
		return;
	}
	if (abs(delta_RD) == 0.0) {
		if (delta_LD > 0) d_g_ang1[y][x] = 3;
		else if (delta_LD < 0) d_g_ang1[y][x] = 7;
		else d_g_ang1[y][x] = -1;
		return;
	}
	angle = atan2(delta_LD, delta_RD);
	if (angle * 8.0 > 7.0 * PI) { d_g_ang1[y][x] = 5; return; }
	if (angle * 8.0 > 5.0 * PI) { d_g_ang1[y][x] = 4; return; }
	if (angle * 8.0 > 3.0 * PI) { d_g_ang1[y][x] = 3; return; }
	if (angle * 8.0 > 1.0 * PI) { d_g_ang1[y][x] = 2; return; }
	if (angle * 8.0 > -1.0 * PI) { d_g_ang1[y][x] = 1; return; }
	if (angle * 8.0 > -3.0 * PI) { d_g_ang1[y][x] = 0; return; }
	if (angle * 8.0 > -5.0 * PI) { d_g_ang1[y][x] = 7; return; }
	if (angle * 8.0 > -7.0 * PI) { d_g_ang1[y][x] = 6; return; }
	d_g_ang1[y][x] = 5;
}

__global__ void cuda_calculatesHoG()
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((y >= ROW - 4) || (x >= COL - 4)) {
		return;
	}
	double HoG[8];
	for (int dir = 0; dir < 8; dir++)
	{
		HoG[dir] = 0.0;
	}

	for (int i = 0; i < 5;i++)
	{
		for (int j = 0; j < 5; j++)
		{
			int ang = d_g_ang1[y + i][x + j];
			if (ang == -1)
			{
				continue;
			}
			HoG[ang] += d_g_nor1[y + i][x + j];
		}
	}

	int maxIndex = 0;
	int maxIndex2 = 0;
	for (int dir = 1; dir < 8; dir++)
	{
		if (HoG[maxIndex] <= HoG[dir])
		{
			maxIndex = dir;
		}
	}

	for (int dir = 1; dir < 8; dir++)
	{
		if (HoG[maxIndex2] <= HoG[dir] && maxIndex != dir)
		{
			maxIndex2 = dir;
		}
	}

	int sHoG = -1;
	if (HoG[maxIndex] > SHoGTHRE)
	{
		sHoG = maxIndex * 8;
		if (HoG[maxIndex2] > SHoGSECONDTHRE*HoG[maxIndex])
		{
			sHoG += maxIndex2;
		}
	}
	d_sHoG[y][x] = sHoG;
}

void procImageInitial()
{
	gpuErrchk(cudaGetSymbolAddress(&d_image1_ptr, d_image1));
	gpuErrchk(cudaGetSymbolAddress(&d_g_can1_ptr, d_g_can1));
	gpuErrchk(cudaGetSymbolAddress(&d_g_nor1_ptr, d_g_nor1));
	gpuErrchk(cudaGetSymbolAddress(&d_g_ang1_ptr, d_g_ang1));
	gpuErrchk(cudaGetSymbolAddress(&d_sHoG_ptr, d_sHoG));
	gpuErrchk(cudaGetSymbolAddress((void**)&d_cuda_defcan_vars_ptr, d_cuda_defcan_vars));
	gpuErrchk(cudaGetSymbolAddress((void**)&d_cuda_defcan_vars_array_ptr, d_cuda_defcan_vars_array));
	gpuStop()	
}

void cuda_procImg(double* g_can, int* g_ang, double* g_nor, unsigned char* image1,int copy) {

	if(copy == 1)
	cudaMemcpy(d_image1_ptr, image1, ROW*COL * sizeof(unsigned char), cudaMemcpyHostToDevice);


	setGPUSize(COL,ROW,TPB,TPB);
	cudaMemset(d_cuda_defcan_vars_ptr, 0, 3 * sizeof(double));
	cuda_defcan1 << <numBlock, numThread >> > ();

	int size = iDivUp(COL*ROW, TPB_X_TPB);
	cuda_sumMatrix << <size, TPB_X_TPB >> > (d_cuda_defcan_vars_array_ptr, d_cuda_defcan_vars_ptr, ROW, COL, 0);
	cuda_sumMatrix << <size, TPB_X_TPB >> > (d_cuda_defcan_vars_array_ptr, d_cuda_defcan_vars_ptr, ROW, COL, 1);
	cuda_sumMatrix << <size, TPB_X_TPB >> > (d_cuda_defcan_vars_array_ptr, d_cuda_defcan_vars_ptr, ROW, COL, 2);

	cuda_defcan2 << <numBlock, numThread >> > ();
	cuda_roberts8 << <numBlock, numThread >> > ();
	cuda_calculatesHoG << <numBlock, numThread >> > ();
	//cudaMemcpy(g_can, d_g_can1_ptr, ROW*COL * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(g_ang, d_g_ang1_ptr, ROW*COL * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(g_nor, d_g_nor1_ptr, ROW*COL * sizeof(double), cudaMemcpyDeviceToHost);
	gpuStop()

}

void getsHoGAndCan(int *sHoG,double *g_can)
{
	if (isGPU == 1)
	{
		cudaMemcpy(sHoG, d_sHoG_ptr, (ROW - 4)*(COL - 4) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(g_can, d_g_can1_ptr, ROW*COL * sizeof(double), cudaMemcpyDeviceToHost);
	}
}
#pragma endregion ProcImage


#pragma region Bilinear

void bilinearInitial()
{
	gpuErrchk(cudaGetSymbolAddress(&d_gpt_ptr, d_gpt));
	gpuErrchk(cudaGetSymbolAddress(&d_image2_ptr, d_image2));
	gpuStop()
}

__global__ void cuda_calc_bilinear_normal_inverse_projection(int x_size1, int y_size1, int x_size2, int y_size2) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((y >= y_size1) || (x >= x_size1)) {
		return;
	}
	int cx, cy, cx2, cy2;
	if (y_size1 == ROW) {
		cx = CX, cy = CY;
		cx2 = CX2, cy2 = CY2;
	}
	else {
		cx = CX2, cy = CY2;
		cx2 = CX, cy2 = CY;
	}

	double inVect[3], outVect[3];
	double x_new, y_new, x_frac, y_frac;
	double gray_new;
	int m, n;

	inVect[2] = 1.0;
	inVect[1] = y - cy;
	inVect[0] = x - cx;

	int i, j;
	double sum;
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
		gray_new = (1.0 - y_frac) * ((1.0 - x_frac) * d_image2[n][m] + x_frac * d_image2[n][m + 1])
			+ y_frac * ((1.0 - x_frac) * d_image2[n + 1][m] + x_frac * d_image2[n + 1][m + 1]);
		d_image1[y][x] = (unsigned char)gray_new;
	}
	else {
#ifdef BACKGBLACK
		d_image1[y][x] = BLACK;
#else
		d_image1[y][x] = WHITE;
#endif
	}
}

void cuda_bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,unsigned char* image1,unsigned char* image2 , int initial) {
	/* inverse projection transformation of the image by bilinear interpolation */

	if (initial == 1)
		cudaMemcpy(d_image2_ptr, image1, sizeof(unsigned char)*ROW2*COL2, cudaMemcpyHostToDevice);


	cudaMemcpy(d_gpt_ptr,gpt, 3 * 3 * sizeof(double), cudaMemcpyHostToDevice);
	setGPUSize(COL, ROW, TPB, TPB);
	cuda_calc_bilinear_normal_inverse_projection << <numBlock, numThread >> > (x_size1, y_size1, x_size2, y_size2);
}

#pragma endregion Bilinear


#pragma region SHoGPat



__device__ int d_dnnL[] = DNNL;
__device__ double d_dnn[4];
__device__ double d_dnn_array[(ROW - 4)*(COL - 4)];
double *d_dnn_ptr;
double *d_dnn_array_ptr;

__device__ int d_count[4];
__device__ int d_count_array[(ROW - 4)*(COL - 4)];
int *d_count_ptr;
int *d_count_array_ptr;

void sHoGpatInitial(int *inteAng)
{
	gpuErrchk(cudaGetSymbolAddress(&d_inteAng_ptr, d_inteAng));
	gpuErrchk(cudaGetSymbolAddress((void**)&d_dnn_ptr, d_dnn));
	gpuErrchk(cudaGetSymbolAddress((void**)&d_dnn_array_ptr, d_dnn_array));


	gpuErrchk(cudaGetSymbolAddress((void**)&d_count_ptr, d_count));
	gpuErrchk(cudaGetSymbolAddress((void**)&d_count_array_ptr, d_count_array));


	cudaMemcpy(d_inteAng_ptr, inteAng, ROWINTE*COLINTE * 64 * sizeof(int), cudaMemcpyHostToDevice);
	gpuStop()
}

__global__ void cuda_sHoGpatInte(int nDnnL)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= COL - 4 || y >= ROW - 4)
	{
		return;
	}
	int maxWinP = MAXWINDOWSIZE + 1;
	int x1 = x + 2;
	int y1 = y + 2;
	int ang1 = 0;
	int count = 0;
	double dnn = 0;
	if ((ang1 = d_sHoG[y][x]) != -1)
	{
		for (int wN = 0; wN < nDnnL; wN++)
		{
			int pPos = maxWinP + d_dnnL[wN];
			int mPos = MAXWINDOWSIZE - d_dnnL[wN];
			double secInte = d_inteAng[ang1][y1 + pPos][x1 + pPos]
				- d_inteAng[ang1][(y1 + pPos)][x1 + mPos]
				- d_inteAng[ang1][(y1 + mPos)][x1 + pPos]
				+ d_inteAng[ang1][(y1 + mPos)][x1 + mPos];
			if (secInte > 0)
			{
				count = 1;
				dnn = d_dnnL[wN];
				break;
			}
		}
	}
	int location = y * (COL - 4) + x;
	d_dnn_array[location] = dnn;
	d_count_array[location] = count;
}



double sHoGpatInteGPU()
{
	cudaMemset(d_count_ptr, 0, sizeof(int));
	cudaMemset(d_dnn_ptr, 0, sizeof(double));
	int dnnL[] = DNNL;
	int nDnnL = NDNNL;
	int count = 0;
	double dnn = 0;
	for (int wN = 0; wN < NDNNL; ++wN)
	{
		if (dnnL[wN] >= MAXWINDOWSIZE)
		{
			nDnnL = wN + 1;
			break;
		}
	}

	setGPUSize(COL, ROW, TPB, TPB);
	cuda_sHoGpatInte << <numBlock, numThread >> > (nDnnL);

	int size = iDivUp((COL-4)*(ROW-4), TPB_X_TPB);
	cuda_sumMatrix << <size, TPB_X_TPB >> > (d_dnn_array_ptr, d_dnn_ptr, ROW-4, COL-4, 0);
	cuda_sumMatrix << <size, TPB_X_TPB >> > (d_count_array_ptr, d_count_ptr, ROW-4, COL-4, 0);


	cudaMemcpy(&count, d_count_ptr, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dnn, d_dnn_ptr, sizeof(double), cudaMemcpyDeviceToHost);

	double ddnn;
	if (count == 0)
		ddnn = MAXWINDOWSIZE;
	else
		ddnn = (double)dnn / count;
	return ddnn;
}

#pragma endregion SHoGPat

#pragma region SHoGCore

//__device__ double d_g_sum_arrary[27*(ROW - 4)*(COL - 4)];
__device__ double d_matrixSum[32];

void *d_matrixSum_ptr;
double *matrixSum;

void gptcorsHoGInteInitial(double *inteCanDir, double *inteDx2Dir, double *inteDy2Dir)
{
	matrixSum = new double[27];
	gpuErrchk(cudaGetSymbolAddress(&d_matrixSum_ptr, d_matrixSum));
	gpuErrchk(cudaGetSymbolAddress(&d_inteCanDir_ptr, d_inteCanDir));
	gpuErrchk(cudaGetSymbolAddress(&d_inteDx2Dir_ptr, d_inteDx2Dir));
	gpuErrchk(cudaGetSymbolAddress(&d_inteDy2Dir_ptr, d_inteDy2Dir));
	cudaMemcpy(d_inteCanDir_ptr, inteCanDir, ROWINTE*COLINTE * 64 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inteDx2Dir_ptr, inteDx2Dir, ROWINTE*COLINTE * 64 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inteDy2Dir_ptr, inteDy2Dir, ROWINTE*COLINTE * 64 * sizeof(double), cudaMemcpyHostToDevice);
	gpuStop()
}

__global__ void cuda_gptcorsHoGInte(double dnn, int pPos, int mPos)
{
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int x1 = blockIdx.x*blockDim.x + threadIdx.x + 2;
	int y1 = blockIdx.y*blockDim.y + threadIdx.y + 2;

	__shared__ double sdata[TPB_X_TPB];
	sdata[tid] = 0;

	if (x1 >= COL - 2 || y1 >= ROW - 2)
	{
		return;
	}
	int dy1 = y1 - CY;
	int dx1 = x1 - CX;
	double t0 = 0, tx2 = 0, ty2 = 0;


	int ang;
	if ((ang = d_sHoG[y1 - 2][x1 - 2]) != -1)
	{
		t0 = d_inteCanDir[ang][y1 + pPos][x1 + pPos] - d_inteCanDir[ang][y1 + pPos][x1 + mPos]
			- d_inteCanDir[ang][y1 + mPos][x1 + pPos] + d_inteCanDir[ang][y1 + mPos][x1 + mPos];
		tx2 = d_inteDx2Dir[ang][y1 + pPos][x1 + pPos] - d_inteDx2Dir[ang][y1 + pPos][x1 + mPos]
			- d_inteDx2Dir[ang][y1 + mPos][x1 + pPos] + d_inteDx2Dir[ang][y1 + mPos][x1 + mPos];
		ty2 = d_inteDy2Dir[ang][y1 + pPos][x1 + pPos] - d_inteDy2Dir[ang][y1 + pPos][x1 + mPos]
			- d_inteDy2Dir[ang][y1 + mPos][x1 + pPos] + d_inteDy2Dir[ang][y1 + mPos][x1 + mPos];
		double image = d_g_can1[y1][x1];
		t0 *= image;
		tx2 *= image;
		ty2 *= image;
	}

	double tx1, tx1x1, tx1x1x1;
	double tx2x1, ty2x1, tx2y1, ty2y1;


	sdata[tid] = tx1 = t0 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 1);

	sdata[tid] = tx1x1 = tx1 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 3);

	sdata[tid] = tx1x1x1 = tx1x1 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 6);

	sdata[tid] = tx1x1x1 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 10);

	sdata[tid] = tx1x1x1 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 11);



	//g0
	sdata[tid] = t0;
	__syncthreads();
	customAdd(sdata, d_matrixSum);
	//gy1
	t0 *= dy1;
	sdata[tid] = t0;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 2);
	//gy1y1
	t0 *= dy1;
	sdata[tid] = t0;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 5);
	//gy1y1y1
	t0 *= dy1;
	sdata[tid] = t0;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 9);
	//gy1y1y1y1
	t0 *= dy1;
	sdata[tid] = t0;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 14);

	//gx1y1
	tx1 *= dy1;
	sdata[tid] = tx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 4);
	//gx1y1y1
	tx1 *= dy1;
	sdata[tid] = tx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 8);
	//gx1y1y1y1
	tx1 *= dy1;
	sdata[tid] = tx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 13);

	//gx1x1y1
	tx1x1 *= dy1;
	sdata[tid] = tx1x1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 7);
	//gx1x1y1y1
	tx1x1 *= dy1;
	sdata[tid] = tx1x1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 12);



	sdata[tid] = tx2;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 15);

	sdata[tid] = ty2;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 16);

	sdata[tid] = tx2x1 = tx2 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 17);

	sdata[tid] = ty2x1 = ty2 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 18);

	sdata[tid] = tx2y1 = tx2 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 19);

	sdata[tid] = ty2y1 = ty2 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 20);

	sdata[tid] = tx2x1 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 21);

	sdata[tid] = ty2x1 * dx1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 22);

	sdata[tid] = tx2x1 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 23);

	sdata[tid] = ty2x1 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 24);

	sdata[tid] = tx2y1 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 25);

	sdata[tid] = ty2y1 * dy1;
	__syncthreads();
	customAdd(sdata, d_matrixSum + 26);
}

double* gptcorsHoGInteGPU(double dnn) 
{
	int windowS = (int)(WGS * WGT * dnn + 0.9999);
	int pPos, mPos;
	if (windowS > MAXWINDOWSIZE)
		windowS = MAXWINDOWSIZE;
	pPos = MAXWINDOWSIZE + 1 + windowS;
	mPos = MAXWINDOWSIZE - windowS;
	setGPUSize(COL, ROW, TPB, TPB);
	cudaMemset(d_matrixSum_ptr, 0, 27 * sizeof(double));
	cuda_gptcorsHoGInte << <numBlock, numThread >> > (dnn, pPos, mPos);
	cudaMemcpy(matrixSum, d_matrixSum_ptr, 27 * sizeof(double), cudaMemcpyDeviceToHost);

	return matrixSum;
}


#pragma endregion SHoGPat

