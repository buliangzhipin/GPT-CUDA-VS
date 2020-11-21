#include <stdio.h>
#include <iostream>
#include <time.h>


#include "cudaInclude.cuh"
#include "init_cuda.h"

#pragma region DeviceMemoryPointer
void *d_cuda_defcan_vars_ptr;
void *d_cuda_defcan_array_ptr;
void *d_image1_ptr;
void *d_g_can1_ptr, *d_g_ang1_ptr, *d_g_nor1_ptr;
#pragma endregion DeviceMemoryPointer

int iDivUp(int hostPtr, int b) { return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); };


dim3 numBlock;
dim3 numThread;

unsigned int nextPow2(unsigned int x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

void setGPUSize(int blockX, int blockY, int threadX, int threadY)
{
	numBlock.x = iDivUp(blockX, threadX);
	numBlock.y = iDivUp(blockY, threadY);
	numThread.x = threadX;
	numThread.y = threadY;
}

template<typename T>
__device__ void customAdd(T* sdata, T* g_odata) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * blockDim.x + tx;
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	// do reduction in shared mem
	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32) { sdata[tid] += sdata[tid + 32]; }__syncthreads();
	if (tid < 16) { sdata[tid] += sdata[tid + 16]; }__syncthreads();
	if (tid < 8) { sdata[tid] += sdata[tid + 8]; }__syncthreads();
	if (tid < 4) { sdata[tid] += sdata[tid + 4]; }__syncthreads();
	if (tid < 2) { sdata[tid] += sdata[tid + 2]; }__syncthreads();
	if (tid < 1) { sdata[tid] += sdata[tid + 1]; }__syncthreads();
	// write result for this block to global mem
	if (tid == 0) 
	{ 
		g_odata[id] = sdata[0];
	}

}

template<typename T>
__device__ void customAdd2(T* sdata, T* g_odata)
{
	int tid = threadIdx.x;
	

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_odata[0] = sdata[0];
}

__global__ void cuda_add2()
{
	customAdd2(d_cuda_defcan_array[0],d_cuda_defcan_vars);
	customAdd2(d_cuda_defcan_array[1],d_cuda_defcan_vars +1);
	customAdd2(d_cuda_defcan_array[2],d_cuda_defcan_vars +2);
}

__global__ void cuda_defcan1() {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tid = ty * blockDim.x + tx;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((y >= ROW) || (x >= COL)) {
		return;
	}

	/* definite canonicalization */
	int margine = CANMARGIN / 2;
	int condition = ((x >= margine && y >= margine) &&
		(x < COL - margine) && (y < ROW - margine) &&
		d_image1[y][x] != WHITE);

	double this_pixel = condition * (double)d_image1[y][x];
	__shared__ double sdata[3][TPB_X_TPB];
	sdata[0][tid] = this_pixel;
	sdata[1][tid] = this_pixel * this_pixel;
	sdata[2][tid] = condition;

	__syncthreads();

	customAdd(sdata[0], d_cuda_defcan_array[0]);
	customAdd(sdata[1], d_cuda_defcan_array[1]);
	customAdd(sdata[2], d_cuda_defcan_array[2]);
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


void cuda_CopyImage()
{

}

void procImageInitial()
{
	gpuErrchk(cudaGetSymbolAddress(&d_image1_ptr, d_image1));
	gpuErrchk(cudaGetSymbolAddress(&d_g_can1_ptr, d_g_can1));
	gpuErrchk(cudaGetSymbolAddress(&d_g_nor1_ptr, d_g_nor1));
	gpuErrchk(cudaGetSymbolAddress(&d_g_ang1_ptr, d_g_ang1));
	gpuErrchk(cudaGetSymbolAddress(&d_cuda_defcan_vars_ptr, d_cuda_defcan_vars));
	gpuErrchk(cudaGetSymbolAddress(&d_cuda_defcan_array_ptr, d_cuda_defcan_array));

	gpuStop()	
}


void cuda_procImg(double* g_can, int* g_ang, double* g_nor, unsigned char* image1,int copy) {
	//cudaMemset(d_cuda_defcan_vars_ptr, 0, 3 * sizeof(double));

	if(copy == 1)
	cudaMemcpy(d_image1_ptr, image1, ROW*COL * sizeof(unsigned char), cudaMemcpyHostToDevice);

	setGPUSize(COL,ROW,TPB,TPB);
	cuda_defcan1 << <numBlock, numThread >> > ();
	//double* cuda_defcan_array = new double[3 * 1024];
	//cudaMemcpy(cuda_defcan_array, d_cuda_defcan_array_ptr, 3 * 1024 * sizeof(double), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < numBlock.x * numBlock.y + 10; i++)
	//{
	//	cout << cuda_defcan_array[i] << " " << cuda_defcan_array[i + 1024] << " " << cuda_defcan_array[i + 2048] << endl;
	//}
	//double num1 = 0;
	//double num2 = 0;
	//double num3 = 0;
	//for (int i = 0; i < numBlock.x * numBlock.y + 10; i++)
	//{
	//	num1 += cuda_defcan_array[i];
	//	num2 += cuda_defcan_array[i+1024];
	//	num3 += cuda_defcan_array[i + 2048];
	//}

	//cout << "sum cpu" << endl;
	//cout << num1 << " " << num2 << " " << num3 << endl;

	unsigned int size = nextPow2(numBlock.x * numBlock.y);

	cuda_add2 << <1, size >> > ();
	cuda_defcan2 << <numBlock, numThread >> > ();
	cuda_roberts8 << <numBlock, numThread >> > ();

	//double* cuda_defcan = new double[3];
	//cudaMemcpy(cuda_defcan, d_cuda_defcan_vars_ptr, 3* sizeof(double), cudaMemcpyDeviceToHost);

	//double* cuda_defcan_array2 = new double[3 * 1024];
	//cudaMemcpy(cuda_defcan_array2, d_cuda_defcan_array_ptr, 3 * 1024 * sizeof(double), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < numBlock.x * numBlock.y + 10; i++)
	//{
	//	cout << cuda_defcan_array2[i] << " " << cuda_defcan_array2[i + 1024] << " " << cuda_defcan_array2[i + 2048] << endl;
	//}

	//cout << "sum" << endl;
	//cout << cuda_defcan[0] << " " << cuda_defcan[1] << " " << cuda_defcan[2] << endl;

	cudaMemcpy(g_can, d_g_can1_ptr, ROW*COL * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(g_ang, d_g_ang1_ptr, ROW*COL * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(g_nor, d_g_nor1_ptr, ROW*COL * sizeof(double), cudaMemcpyDeviceToHost);
	gpuStop()

}

// main function
//int main(void)
//{
//	bool GPU = true;
//
//	int N = 1000000;
//	float *host_x, *host_y, *dev_x, *dev_y;
//
//	// CPU側の領域確保
//	host_x = (float*)malloc(N * sizeof(float));
//	host_y = (float*)malloc(N * sizeof(float));
//
//	// 乱数値を入力する
//	for (int i = 0; i < N; i++) {
//		host_x[i] = rand();
//	}
//
//	int start = clock();
//
//	if (GPU == true) {
//
//		// デバイス(GPU)側の領域確保
//		cudaMalloc(&dev_x, N * sizeof(float));
//		cudaMalloc(&dev_y, N * sizeof(float));
//
//		// CPU⇒GPUのデータコピー
//		cudaMemcpy(dev_x, host_x, N * sizeof(float), cudaMemcpyHostToDevice);
//
//		// GPUで計算
//		gpu_function << <(N + 255) / 256, 256 >> > (dev_x, dev_y);
//
//		// GPU⇒CPUのデータコピー
//		cudaMemcpy(host_y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);
//
//	}
//	else {
//		// CPUで計算
//		cpu_function(N, host_x, host_y);
//	}
//
//	int end = clock();
//
//	// 計算が正しく行われているか確認
//	float sum = 0.0f;
//	for (int j = 0; j < N; j++) {
//		sum += host_y[j];
//	}
//	std::cout << sum << std::endl;
//
//	// 最後に計算時間を表示
//	std::cout << end - start << "[ms]" << std::endl;
//
//	return 0;
//}