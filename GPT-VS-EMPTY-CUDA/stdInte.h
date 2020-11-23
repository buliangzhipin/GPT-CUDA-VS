#pragma once
#include "parameter.h"


#define WNNDEGD 1.2 /* NNDEGDを四角で測っているための補正 */




/*----------------------------------------------------------------------------*/
void copyNormalGpt(double inGpt[3][3], double outGpt[3][3]);
void multplyMV(double inMat[NI][NI + 1], double v[NI]);
void solveLEq(double inMat[NI][NI + 1]);
void initGpt(double gpt[3][3]);

void calInte64(double* g_can, int* sHOG, int* inteAng,
	double* inteCanDir, double* inteDx2Dir, double* inteDy2Dir);
double sHoGpatInte(int* sHoG1, int* inteAng);
void gptcorsHoGInte(int* sHoG1, double* g_can1,
	int* sHoG2, double* g_can2, double* gwt, double* inteCanDir,
	double* inteDx2Dir, double* inteDy2Dir, double dnn, double gpt[3][3]);

