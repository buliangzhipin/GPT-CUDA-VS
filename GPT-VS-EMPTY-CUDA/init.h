#pragma once
#include <cmath>

#include "parameter.h"
#include "utility.h"

void procImg(float* g_can, int* g_ang, float* g_nor, int* sHoG, unsigned char* image1,int initial);
void roberts8(int* g_ang, float* g_nor, unsigned char* image1);
void defcan2(float* g_can, unsigned char* image1);
void smplHoG64(int* sHoG, int* g_ang, float* g_nor);
void bilinear_normal_projection(float gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
	unsigned char* image1, unsigned char* image2,int initial);
void bilinear_normal_inverse_projection(float gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
	unsigned char* image1, unsigned char* image2);


