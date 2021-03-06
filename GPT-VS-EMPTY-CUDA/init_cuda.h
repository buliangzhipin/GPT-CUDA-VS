#pragma once

void procImageInitial();
void cuda_procImg(double* g_can, int* g_ang, double* g_nor, unsigned char* image1, int copy);

void bilinearInitial();
void cuda_bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2, unsigned char* image1, unsigned char* image2, int initial);
void getsHoGAndCan(int *sHoG, double *g_can);
void getImage(unsigned char* image2);

