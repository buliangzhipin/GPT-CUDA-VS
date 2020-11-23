// GPT-VS-EMPTY-CUDA.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#include <iostream>
#include <fstream>
#include <direct.h>
#include<ctime>
using namespace std;
#include "main.h"
#include "init.h"
#include "parameter.h"
#include "utility.h"
#include "init_cuda.h"
#include "stdInte.cuh"

#if isGPU == 0
#define variableChar(a) #a
#elif isGPU == 1
#define variableChar(a) ""#a"CUDA" 
#endif

template<typename T>
void SaveData(T* data,const char* fileName,int x_size,int y_size)
{
	ofstream saveFile(fileName);
	for (int y = 0; y<y_size;y++)
	{
		for (int x = 0;x<x_size;x++)
		{
			saveFile << data[y*x_size + x] << " ";
		}
		saveFile << endl;
	}
	saveFile.close();
}

int main()
{
	double gpt1[3][3];
	initGpt2(gpt1, ZOOM, ZOOM * BETA, B1, B2, ROT);

	procImageInitial();
	bilinearInitial();

#pragma region Load_And_Proc_Image
	unsigned char* image1;
	image1 = new unsigned char[ROW2*COL2];
	char fileName[128];
	sprintf(fileName, "%s/%s.pgm", IMGDIR, RgIMAGE);
	load_image_file(fileName, image1, COL, ROW);

	int *g_ang2 = new int[ROW*COL];	  // direction of gradients
	int *sHoG2 = new int[(ROW - 4)*(COL - 4)];
	double *g_nor2 = new double[ROW*COL]; // norm of gradients
	double *g_can2 = new double[ROW*COL]; // canonicalized images
	procImg(g_can2, g_ang2, g_nor2, sHoG2, image1,1);
	getsHoGAndCan(sHoG2, g_can2);
#pragma endregion Load_And_Proc_Image


	//積分画像計算　CUDA化する必要あります。
#pragma region Calculate_Inte
	int* inteAng = new int[ROWINTE*COLINTE*64];
	double* inteCanDir = new double[ROWINTE*COLINTE*64];
	double* inteDx2Dir = new double[ROWINTE*COLINTE*64];
	double* inteDy2Dir = new double[ROWINTE*COLINTE*64];
	clock_t start1, end1;
	start1 = clock();
	calInte64(g_can2, sHoG2, inteAng, inteCanDir, inteDx2Dir, inteDy2Dir);
	end1 = clock();		//程序结束用时
	double endtime1 = (double)(end1 - start1) / CLOCKS_PER_SEC;
	cout << "Total time Inte:" << endtime1 * 1000 << "ms" << endl;	//ms为单位
#pragma endregion Calculate_Inte

	sHoGpatInitial(inteAng);
	sHoGcoreInitial(inteCanDir, inteDx2Dir, inteDy2Dir);


	cout << "process1 finished" << endl;


	//image2処理
#pragma region Load_And_Proc_Image2
	sprintf(fileName, "%s/%s.pgm", IMGDIR, TsIMAGE);
	load_image_file(fileName, image1, COL2, ROW2);
	unsigned char *image2 = new unsigned char[COL2*ROW2];
	/* save the initial image */
	for (int y = 0; y < ROW2; y++)
		for (int x = 0; x < COL2; x++)
			image2[y*COL2+x] = image1[y*COL2+x];
	bilinear_normal_projection(gpt1, COL, ROW, COL2, ROW2, image1, image2,1);
	sprintf(fileName, "%s/%s_init.pgm", IMGDIR, RgIMAGE);
	//save_image_file(fileName, image2, COL, ROW);

	int *g_ang1 = new int[ROW*COL];	  // direction of gradients
	int *sHoG1 = new int[(ROW - 4)*(COL - 4)];
	double *g_nor1 = new double[ROW*COL]; // norm of gradients
	double *g_can1 = new double[ROW*COL]; // canonicalized images
	procImg(g_can1, g_ang1, g_nor1, sHoG1, image2,0); 
	getsHoGAndCan(sHoG1, g_can1);

#pragma endregion Load_And_Proc_Image
	cout << "process2 finished" << endl;


#pragma region Calculate_Initial_Correlation
	/* calculate the initial correlation */
	double org_cor, gat_corf, gat_corb;
	double old_cor0, old_cor1, new_cor1; //
	old_cor1 = 0.0;
	for (int y = MARGINE; y < ROW - MARGINE; y++)
		for (int x = MARGINE; x < COL - MARGINE; x++)
			old_cor1 += g_can1[y*COL+x] * g_can2[y*COL+x];
	org_cor = old_cor1;
	printf("Original cor. = %f\n", org_cor);
	old_cor0 = old_cor1;
#pragma endregion Calculate_Initial_Correlation
/*
	SaveData<double>(g_can2, variableChar(gCan1), COL, ROW);
	SaveData<double>(g_nor2, variableChar(gNor2), COL, ROW);
	SaveData<int>(g_ang2, variableChar(gAng2), COL, ROW);
	SaveData<double>(g_can1, variableChar(gCan1), COL, ROW);
	SaveData<double>(g_nor1, variableChar(gNor1), COL, ROW);
	SaveData<int>(g_ang1, variableChar(gAng1), COL, ROW);*/
	
	//Initial dnn
	double d2 = 0.0;
	double dnn = WNNDEsHoGD * sHoGpatInte(sHoG1, inteAng);

	double* gwt = new double[ROW*COL];

	cout << dnn << endl;
	cout << "test" << endl;

	//time
		int margine = CANMARGIN / 2;
		clock_t start, end;
		start = clock();
		for (int iter = 0; iter < MAXITER; iter++)
		{
			//Match
			gptcorsHoGInte(sHoG1, g_can1, sHoG2, g_can2, gwt, inteCanDir, inteDx2Dir, inteDy2Dir, dnn, gpt1);
			bilinear_normal_projection(gpt1, COL, ROW, COL2, ROW2, image1, image2,0);
			procImg(g_can1, g_ang1, g_nor1, sHoG1, image2,0);
			dnn = WNNDEsHoGD * sHoGpatInte(sHoG1, inteAng);

			/* update correlation */
			//new_cor1 = 0.0;
			//getsHoGAndCan(sHoG1, g_can1);
			//for (int y = margine; y < ROW - margine; y++)
			//	for (int x = margine; x < COL - margine; x++)
			//		new_cor1 += g_can1[y*COL+x] * g_can2[y*COL+x];		
			//printf("iter = %d, new col. = %f dnn = %f   (d2 = %f) \n", iter, new_cor1, dnn, d2);
		}
		end = clock();		//程序结束用时
		double endtime = (double)(end - start) / CLOCKS_PER_SEC;
		cout << "Total time:" << endtime * 1000 << "ms" << endl;	//ms为单位


		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cout << gpt1[i][j] << endl;
			}
	
		}

		delete image1;
		system("pause");
		return 0;
}

void initGpt2(double gpt[3][3], double alpha, double beta, double b1, double b2, double rotation)
{
	int x, y;
	double tmp[3][3], gpt_[3][3];

	for (x = 0; x < 3; x++)
		for (y = 0; y < 3; y++)
			gpt_[x][y] = 0.0;

	gpt[0][0] = alpha, gpt[0][1] = 0.0;
	gpt[1][0] = 0.0;
	gpt[1][1] = beta;
	gpt[0][2] = b1;
	gpt[1][2] = b2;
	gpt[2][0] = 0.0;
	gpt[2][1] = 0.0;
	gpt[2][2] = 1.0;

	tmp[0][0] = cos(rotation * PI / 180.0), tmp[0][1] = sin(rotation * PI / 180.0);
	tmp[1][0] = -sin(rotation * PI / 180.0), tmp[1][1] = cos(rotation * PI / 180.0);
	tmp[0][2] = tmp[1][2] = tmp[2][0] = tmp[2][1] = 0.0;
	tmp[2][2] = 1.0;

	multiply3x3(tmp, gpt, gpt_);
	inverse3x3(gpt_, gpt);
}
