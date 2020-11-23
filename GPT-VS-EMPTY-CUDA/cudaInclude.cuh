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


#pragma region Parameter
#define TPB 32
#define TPB_X_TPB TPB*TPB
#define G_NUM 30
#pragma endregion Parameter
