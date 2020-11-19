#pragma once

#include <iostream>
using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "parameter.h"


#pragma region MACRO
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		cout << stderr << "GPUassert: "<< file << cudaGetErrorString(code)<< line <<"\n"<< endl;
		if (abort) exit(code);
	}
}

#define gpuStop() gpuErrchk(cudaDeviceSynchronize());\
	gpuErrchk(cudaPeekAtLastError()); // Checks for launch error
//gpuErrchk(cudaThreadSynchronize()); // Checks for execution error 已否决
#pragma endregion MACRO

#pragma region DeviceMemory
__device__ double d_cuda_defcan_vars[3];
__device__ unsigned char d_image1[ROW][COL];
__device__ unsigned char d_image2[ROW2][COL2];


__device__ double d_g_can1[ROW][COL], d_g_nor1[ROW][COL];
__device__ int d_g_ang1[ROW][COL];
//d_gk[ROW][COL], d_gwt[ROW][COL], d_g_can2[ROW][COL];
#pragma endregion DeviceMemory



#pragma region Parameter
#define TPB 32
#define TPB_X_TPB TPB*TPB
#define G_NUM 30

#pragma endregion Parameter

