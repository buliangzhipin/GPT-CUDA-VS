#pragma once
#include "parameter.h"

#define MU 1.0  /* 緩和係数 of Newton */
#define WGS 1.5 /* Gauss型窓関数を1つの矩形で表すための幅の比 */

#define WNNDEsHoGD  0.7          /* NNDEGDを四角で測っているための補正 */
#define MAXWINDOWSIZE 16
#define WNNDEGD 1.2 /* NNDEGDを四角で測っているための補正 */
#define DNNL {0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512};
#define NDNNL 19
#define ROWINTE (ROW + 2 * MAXWINDOWSIZE + 1)
#define COLINTE (COL + 2 * MAXWINDOWSIZE + 1)
#define NI 8 /* For inverse matrix */

/*----------------------------------------------------------------------------*/
void copyNormalGpt(double inGpt[3][3], double outGpt[3][3]);
void multplyMV(double inMat[NI][NI + 1], double v[NI]);
void solveLEq(double inMat[NI][NI + 1]);
void initGpt(double gpt[3][3]);

void calInte64(double* g_can, char* sHOG, int* inteAng,
	double* inteCanDir, double* inteDx2Dir, double* inteDy2Dir);
double sHoGpatInte(char* sHoG1, int* inteAng);
void gptcorsHoGInte(char* sHoG1, double* g_can1,
	char* sHoG2, double* g_can2, double* gwt, double* inteCanDir,
	double* inteDx2Dir, double* inteDy2Dir, double dnn, double gpt[3][3]);

