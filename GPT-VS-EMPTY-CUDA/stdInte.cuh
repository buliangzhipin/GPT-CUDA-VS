#pragma once

void sHoGpatInitial(int *inteAng);
double sHoGpatInteGPU();

void gptcorsHoGInteInitial(double *inteCanDir, double *inteDx2Dir, double *inteDy2Dir);
double* gptcorsHoGInteGPU(double dnn);

