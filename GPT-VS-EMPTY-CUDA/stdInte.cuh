#pragma once

void sHoGpatInitial(int *inteAng);
double sHoGpatInteGPU(int* sHoG1);

double* gptcorsHoGInteGPU(double dnn);

void sHoGcoreInitial(double *inteCanDir, double *inteDx2Dir, double *inteDy2Dir);
