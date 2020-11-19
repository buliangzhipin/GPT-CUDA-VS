#pragma once
#include <cmath>

#include "parameter.h"
#include "utility.h"

void procImg(double* g_can, int* g_ang, double* g_nor, char* sHoG, unsigned char* image1);
void roberts8(int* g_ang, double* g_nor, unsigned char* image1);
void defcan2(double* g_can, unsigned char* image1);
void smplHoG64(char* sHoG, int* g_ang, double* g_nor);
void bilinear_normal_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
	unsigned char* image1, unsigned char* image2);
void bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
	unsigned char* image1, unsigned char* image2);


