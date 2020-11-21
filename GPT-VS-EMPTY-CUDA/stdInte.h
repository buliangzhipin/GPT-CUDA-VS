#pragma once
#include "parameter.h"

#define MU 1.0  /* 緩和係数 of Newton */
#define WGS 1.5 /* Gauss型窓関数を1つの矩形で表すための幅の比 */

#define WNNDEsHoGD  0.7          /* NNDEGDを四角で測っているための補正 */
#define WNNDEGD 1.2 /* NNDEGDを四角で測っているための補正 */


#define NI 8 /* For inverse matrix */

/*----------------------------------------------------------------------------*/
void copyNormalGpt(float inGpt[3][3], float outGpt[3][3]);
void multplyMV(float inMat[NI][NI + 1], float v[NI]);
void solveLEq(float inMat[NI][NI + 1]);
void initGpt(float gpt[3][3]);

void calInte64(float* g_can, int* sHOG, int* inteAng,
	float* inteCanDir, float* inteDx2Dir, float* inteDy2Dir);
float sHoGpatInte(int* sHoG1, int* inteAng);
void gptcorsHoGInte(int* sHoG1, float* g_can1,
	int* sHoG2, float* g_can2, float* gwt, float* inteCanDir,
	float* inteDx2Dir, float* inteDy2Dir, float dnn, float gpt[3][3]);

