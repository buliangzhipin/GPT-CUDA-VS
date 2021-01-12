#pragma once

void load_image_file(char *filename, unsigned char* image1, int x_size1, int y_size1); /* image input */
void save_image_file(char *, unsigned char* image2, int x_size2, int y_size2);


void multiplyVect3x3(double gpt[3][3], double inVect[3], double outVect[3]);
void multiplyVect4x4(double inMat[4][4], double inVect[4], double outVect[4]);
void multiplyVect8x8(double inMat[8][8], double inVect[8], double outVect[8]);
void multiply3x3(double inMat1[3][3], double inMat2[3][3], double outMat[3][3]);
void inverse3x3(double inMat[3][3], double outMat[3][3]);
void inverse4x4(double inMat[4][4], double outMat[4][4]);
void inverse8x8(double inMat[8][8], double outMat[8][8]);

template<typename  T> void changeValue(T* t1, T* t2) {
	T tmpT;
	tmpT = *t1;
	*t1 = *t2;
	*t2 = tmpT;
}