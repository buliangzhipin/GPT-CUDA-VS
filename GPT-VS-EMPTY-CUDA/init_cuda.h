#pragma once


void cuda_procImg(float* g_can, int* g_ang, float* g_nor, unsigned char* image1, int copy);
void cuda_bilinear_normal_inverse_projection(float gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2, unsigned char* image1, unsigned char* image2, int initial);
void procImageInitial();
void bilinearInitial();
