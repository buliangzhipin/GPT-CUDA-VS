#pragma once
#include "parameter.h"

#define MU 1.0  /* 緩和係数 of Newton */
#define WNNDEsHoGD  0.7          /* NNDEGDを四角で測っているための補正 */
#define WNNDEGD 1.2 /* NNDEGDを四角で測っているための補正 */
#define NI 8 /* For inverse matrix */

/*----------------------------------------------------------------------------*/
void calInte64(double* g_can, int* sHOG, int* inteAng,
	double* inteCanDir, double* inteDx2Dir, double* inteDy2Dir);
double sHoGpatInte(int* sHoG1, int* inteAng);
void gptcorsHoGInte(int* sHoG1, double* g_can1, double* inteCanDir,
	double* inteDx2Dir, double* inteDy2Dir, double dnn, double gpt[3][3]);

