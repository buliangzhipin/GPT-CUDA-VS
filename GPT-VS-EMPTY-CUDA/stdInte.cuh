#pragma once

double sHoGpatInteGPU(int* sHoG1);
void sHoGpatInitial(int *inteAng);

void sHoGcoreInitial(double *inteCanDir, double *inteDx2Dir, double *inteDy2Dir);
void gptcorsHoGInteGPU(double dnn, double gpt[3][3]);

