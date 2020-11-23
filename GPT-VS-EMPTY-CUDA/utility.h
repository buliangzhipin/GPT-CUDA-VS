#pragma once

void load_image_file(char *filename, unsigned char* image1, int x_size1, int y_size1); /* image input */

void multiplyVect3x3(double gpt[3][3], double inVect[3], double outVect[3]);
void multiply3x3(double inMat1[3][3], double inMat2[3][3], double outMat[3][3]);
void inverse3x3(double inMat[3][3], double outMat[3][3]);

template<typename  T> void changeValue(T* t1, T* t2) {
	T tmpT;
	tmpT = *t1;
	*t1 = *t2;
	*t2 = tmpT;
}