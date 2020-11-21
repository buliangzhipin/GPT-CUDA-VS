#pragma once

void load_image_file(char *filename, unsigned char* image1, int x_size1, int y_size1); /* image input */


void multiplyVect3x3(float gpt[3][3], float inVect[3], float outVect[3]);
void multiplyVect4x4(float inMat[4][4], float inVect[4], float outVect[4]);
void multiplyVect8x8(float inMat[8][8], float inVect[8], float outVect[8]);
void multiply3x3(float inMat1[3][3], float inMat2[3][3], float outMat[3][3]);
void inverse3x3(float inMat[3][3], float outMat[3][3]);
void inverse4x4(float inMat[4][4], float outMat[4][4]);
void inverse8x8(float inMat[8][8], float outMat[8][8]);

template<typename  T> void changeValue(T* t1, T* t2) {
	T tmpT;
	tmpT = *t1;
	*t1 = *t2;
	*t2 = tmpT;
}