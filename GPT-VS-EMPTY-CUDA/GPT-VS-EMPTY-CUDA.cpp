// GPT-VS-EMPTY-CUDA.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#include <iostream>
#include <fstream>
#include <direct.h>
using namespace std;
#include "main.h"
#include "init.h"
#include "parameter.h"
#include "utility.h"

int main()
{
	double gpt1[3][3];
	initGpt2(gpt1, ZOOM, ZOOM * BETA, B1, B2, ROT);


	//char* directory;
	//if ((directory = _getcwd(NULL, 0)) == NULL)
	//{
	//	perror("getcwd error");
	//}
	//else
	//{
	//	printf("%s\n", directory);
	//	free(directory);
	//}



#pragma region Initial_Gauss_Function
	double gk[ROW][COL];
	for (int y = 0; y < ROW; y++)
		for (int x = 0; x < COL; x++)
			gk[y][x] = exp(-(x * x + y * y) / 2.0);
#pragma endregion Initial_Gauss_Function

#pragma region Load_And_Proc_Image
	unsigned char* image1;
	image1 = new unsigned char[ROW2*COL2];
	char fileName[128];
	sprintf(fileName, "%s/%s.pgm", IMGDIR, RgIMAGE);
	load_image_file(fileName, image1, COL, ROW);

	int *g_ang2 = new int[ROW*COL];	  // direction of gradients
	char *g_HoG2 = new char[ROW*COL*8]; // HoG feature of the images
	char *sHoG2 = new char[(ROW - 4)*(COL - 4)];
	double *g_nor2 = new double[ROW*COL]; // norm of gradients
	double *g_can2 = new double[ROW*COL]; // canonicalized images
	procImg(g_can2, g_ang2, g_nor2, g_HoG2, sHoG2, image1);
#pragma endregion Load_And_Proc_Image


	//積分画像計算　CUDA化する必要あります。
#pragma region Calculate_Inte
	int* inteAng = new int[ROWINTE*COLINTE*9];
	double* inteCanDir = new double[ROWINTE*COLINTE*9];
	double* inteDx2Dir = new double[ROWINTE*COLINTE*9];
	double* inteDy2Dir = new double[ROWINTE*COLINTE*9];
	calInte(g_can2, g_ang2, inteAng, inteCanDir, inteDx2Dir, inteDy2Dir);
#pragma endregion Calculate_Inte
	ofstream inteAngFile("inteAngWindows.txt");
	ofstream gAngFile("gAngWindows.txt");

	for (int y = 0; y < ROWINTE; y++)
	{
		for (int x = 0; x < COLINTE; x++)
		{
			for (int d = 0; d < 9; d++)
			{
				inteAngFile << inteAng[y*COLINTE + x + d * ROWINTE*COLINTE] << " ";
			}
		}
		inteAngFile << endl;
	}
	for (int y = 0; y < ROW; y++)
	{
		for (int x = 0; x < COL; x++)
		{
			gAngFile << g_ang2[y*COL + x] << " ";
		}
		gAngFile << endl;
	}
	gAngFile.close();
	inteAngFile.close();



	//image2処理
#pragma region Load_And_Proc_Image2
	sprintf(fileName, "%s/%s.pgm", IMGDIR, TsIMAGE);
	load_image_file(fileName, image1, COL2, ROW2);
	unsigned char *image2 = new unsigned char[COL2*ROW2];
	unsigned char *image3 = new unsigned char[COL2*ROW2];
	for (int y = 0; y < ROW2; y++)
		for (int x = 0; x < COL2; x++)
			image3[y*COL2+x] = image1[y*COL2+x];

	/* save the initial image */
	for (int y = 0; y < ROW2; y++)
		for (int x = 0; x < COL2; x++)
			image2[y*COL2+x] = image1[y*COL2+x];
	bilinear_normal_projection(gpt1, COL, ROW, COL2, ROW2, image1, image2);
	sprintf(fileName, "%s/%s_init.pgm", IMGDIR, RgIMAGE);
	//save_image_file(fileName, image2, COL, ROW);

	int *g_ang1 = new int[ROW*COL];	  // direction of gradients
	char *g_HoG1 = new char[ROW*COL * 8]; // HoG feature of the images
	char *sHoG1 = new char[(ROW - 4)*(COL - 4)];
	double *g_nor1 = new double[ROW*COL]; // norm of gradients
	double *g_can1 = new double[ROW*COL]; // canonicalized images
	procImg(g_can1, g_ang1, g_nor1, g_HoG1, sHoG1, image2); 
#pragma endregion Load_And_Proc_Image
	//ofstream outputfile("text.txt");

	//for (int y = 0; y < ROW; y++)
	//{
	//	for (int x = 0; x < COL; x++)
	//	{
	//		outputfile << g_ang1[y*COL + x] << " ";
	//	}
	//	outputfile << endl;
	//}
	//outputfile.close();

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

	//Initial dnn
	double d2 = 0.0;
	double dnn = WNNDEGD * winpatInte(g_ang1, inteAng);
	double* gwt = new double[ROW*COL];

	cout << dnn << endl;

	//time
	int margine = CANMARGIN / 2;
	for (int iter = 0; iter < MAXITER; iter++)
	{
		//update gauss window function
		double var = pow(WGT * dnn, 2);
		for (int y = 0; y < ROW; y++)
			for (int x = 0; x < COL; x++)
				gwt[y*COL+x] = pow(gk[y][x], 1.0 / var);

		//Match
		gptcorInte(g_ang1, g_can1, g_ang2, g_can2, gwt, inteCanDir, inteDx2Dir, inteDy2Dir, dnn, gpt1);
		/* transform the test image and update g_can1, g_ang1, g_nor1, g_HoG1, sHoG1 */
		for (int y = 0; y < ROW2; y++)
			for (int x = 0; x < COL2; x++)
				image1[y*COL2+x] = (unsigned char)image3[y*COL2+x];
		bilinear_normal_projection(gpt1, COL, ROW, COL2, ROW2, image1, image2);
		procImg(g_can1, g_ang1, g_nor1, g_HoG1, sHoG1, image2);

		/* update correlation */
		new_cor1 = 0.0;
		for (int y = margine; y < ROW - margine; y++)
			for (int x = margine; x < COL - margine; x++)
				new_cor1 += g_can1[y*COL+x] * g_can2[y*COL+x];

		dnn = WNNDEGD * winpatInte(g_ang1, inteAng);
		printf("iter = %d, new col. = %f dnn = %f  var = %f (d2 = %f) \n", iter, new_cor1, dnn, 1 / var, d2);
	}


	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << gpt1[i][j] << endl;
		}
	
	}

	delete image1;
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
