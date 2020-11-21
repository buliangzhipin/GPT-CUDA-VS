#include <math.h>
#include <iostream>
using namespace std;
#include "parameter.h"
#include "utility.h"

void multiplyVect3x3(float inMat[3][3], float inVect[3], float outVect[3])
{
	int i, j;
	float sum;
	for (i = 0; i < 3; ++i)
	{
		sum = 0.0;
		for (j = 0; j < 3; ++j)
		{
			sum += inMat[i][j] * inVect[j];
		}
		outVect[i] = sum;
	}
}

void multiplyVect4x4(float inMat[4][4], float inVect[4], float outVect[4])
{
	int i, j;
	float sum;
	for (i = 0; i < 4; ++i)
	{
		sum = 0.0;
		for (j = 0; j < 4; ++j)
		{
			sum += inMat[i][j] * inVect[j];
		}
		outVect[i] = sum;
	}
}

void multiplyVect8x8(float inMat[8][8], float inVect[8], float outVect[8])
{
	int i, j;
	float sum;
	for (i = 0; i < 8; ++i)
	{
		sum = 0.0;
		for (j = 0; j < 8; ++j)
		{
			sum += inMat[i][j] * inVect[j];
		}
		outVect[i] = sum;
	}
}

void multiply3x3(float inMat1[3][3], float inMat2[3][3], float outMat[3][3])
{
	int i, j, k;
	float sum;
	for (i = 0; i < 3; ++i)
	{
		for (j = 0; j < 3; ++j)
		{
			sum = 0.0;
			for (k = 0; k < 3; ++k)
			{
				sum += inMat1[i][k] * inMat2[k][j];
			}
			outMat[i][j] = sum;
		}
	}
}

void multiply8x8(float inMat1[8][8], float inMat2[8][8], float outMat[8][8])
{
	int i, j, k;
	float sum;
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			sum = 0.0;
			for (k = 0; k < 8; ++k)
			{
				sum += inMat1[i][k] * inMat2[k][j];
			}
			outMat[i][j] = sum;
		}
	}
}

void inverse3x3(float inMat[3][3], float outMat[3][3])
{
	float det;
	det = inMat[0][0] * inMat[1][1] * inMat[2][2] + inMat[1][0] * inMat[2][1] * inMat[0][2] + inMat[2][0] * inMat[0][1] * inMat[1][2] - inMat[0][0] * inMat[2][1] * inMat[1][2] - inMat[1][0] * inMat[0][1] * inMat[2][2] - inMat[2][0] * inMat[1][1] * inMat[0][2];
	if (fabs(det) < EPS)
	{
		cout << "3 x 3 matrix has no inverse" << endl;
		exit(1);
	}
	outMat[0][0] = (inMat[1][1] * inMat[2][2] - inMat[2][1] * inMat[1][2]) / det;
	outMat[1][0] = -(inMat[1][0] * inMat[2][2] - inMat[2][0] * inMat[1][2]) / det;
	outMat[2][0] = (inMat[1][0] * inMat[2][1] - inMat[2][0] * inMat[1][1]) / det;

	outMat[0][1] = -(inMat[0][1] * inMat[2][2] - inMat[2][1] * inMat[0][2]) / det;
	outMat[1][1] = (inMat[0][0] * inMat[2][2] - inMat[2][0] * inMat[0][2]) / det;
	outMat[2][1] = -(inMat[0][0] * inMat[2][1] - inMat[2][0] * inMat[0][1]) / det;

	outMat[0][2] = (inMat[0][1] * inMat[1][2] - inMat[1][1] * inMat[0][2]) / det;
	outMat[1][2] = -(inMat[0][0] * inMat[1][2] - inMat[1][0] * inMat[0][2]) / det;
	outMat[2][2] = (inMat[0][0] * inMat[1][1] - inMat[1][0] * inMat[0][1]) / det;
}

void inverse4x4(float inMat[4][4], float outMat[4][4])
{
	float inMemo[4][4];
	float buf;
	int i, j, k;
	int n = 4;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			outMat[i][j] = (i == j) ? 1.0 : 0.0;
			inMemo[i][j] = inMat[i][j];
		}
	}

	for (i = 0; i < n; i++)
	{
		buf = 1 / inMemo[i][i];
		for (j = 0; j < n; j++)
		{
			inMemo[i][j] *= buf;
			outMat[i][j] *= buf;
		}
		for (j = 0; j < n; j++)
		{
			if (i != j)
			{
				buf = inMemo[j][i];
				for (k = 0; k < n; k++)
				{
					inMemo[j][k] -= inMemo[i][k] * buf;
					outMat[j][k] -= outMat[i][k] * buf;
				}
			}
		}
	}
}

void inverse8x8(float inMat[8][8], float outMat[8][8])
{
	float inMemo[8][8];
	float buf;
	int i, j, k;
	int n = 8;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			outMat[i][j] = (i == j) ? 1.0 : 0.0;
			inMemo[i][j] = inMat[i][j];
		}
	}

	for (i = 0; i < n; i++)
	{
		buf = 1 / inMemo[i][i];
		for (j = 0; j < n; j++)
		{
			inMemo[i][j] *= buf;
			outMat[i][j] *= buf;
		}
		for (j = 0; j < n; j++)
		{
			if (i != j)
			{
				buf = inMemo[j][i];
				for (k = 0; k < n; k++)
				{
					inMemo[j][k] -= inMemo[i][k] * buf;
					outMat[j][k] -= outMat[i][k] * buf;
				}
			}
		}
	}
}

void load_image_file(char *filename, unsigned char* image1, int x_size1, int y_size1)
{
	/* Input of header & body information of pgm file */
	/* for image1[ ][ ] */
	char buffer[MAX_BUFFERSIZE];
	FILE *fp;     /* File pointer */
	int max_gray; /* Maximum gray level */
	int x, y;     /* Loop variable */
	/* Input file open */
	fp = fopen(filename, "rb");
	if (NULL == fp)
	{
		printf("     The file doesn't exist! : %s \n\n", filename);
		exit(1);
	}
	/* Check of file-type ---P5 */
	fgets(buffer, MAX_BUFFERSIZE, fp);
	if (buffer[0] != 'P' || buffer[1] != '5')
	{
		printf("     Mistaken file format, not P5!\n\n");
		exit(1);
	}
	/* input of x_size1, y_size1 */
	x_size1 = 0;
	y_size1 = 0;
	while (x_size1 == 0 || y_size1 == 0)
	{
		fgets(buffer, MAX_BUFFERSIZE, fp);
		if (buffer[0] != '#')
		{
			sscanf(buffer, "%d %d", &x_size1, &y_size1);
		}
	}
	//printf("xsize = %d, ysize = %d", x_size1, y_size1);
	/* input of max_gray */
	max_gray = 0;
	while (max_gray == 0)
	{
		fgets(buffer, MAX_BUFFERSIZE, fp);
		if (buffer[0] != '#')
		{
			sscanf(buffer, "%d", &max_gray);
		}
	}
	//????
	//if (x_size1 > ROW2 || y_size1 > COL2)
	//{
	//	printf("     Image size exceeds %d x %d\n\n",
	//		ROW2, COL2);
	//	printf("     Please use smaller images!\n\n");
	//	exit(1);
	//}
	if (max_gray != MAX_BRIGHTNESS)
	{
		printf("     Invalid value of maximum gray level!\n\n");
		exit(1);
	}
	/* Input of image data*/
	for (y = 0; y < y_size1; y++)
	{
		for (x = 0; x < x_size1; x++)
		{
			image1[y*x_size1+x] = (unsigned char)fgetc(fp);
		}
	}
	fclose(fp);
}