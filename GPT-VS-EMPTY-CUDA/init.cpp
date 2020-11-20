#include <stdio.h>

#include "init.h"
#include "init_cuda.h"

void procImg(double* g_can, int* g_ang, double* g_nor, char* sHoG, unsigned char* image1,int initial)
{
#if isGPU == 0
	defcan2(g_can, image1);         /* canonicalization */
	roberts8(g_ang, g_nor, image1); /* 8-quantization of gradient dir */
	// calHoG(g_ang, g_HoG);				/* calculate sHOG pattern */
	smplHoG64(sHoG, g_ang, g_nor); /* Numberring the sHOG pattern to sHoGNUMBER */
#elif isGPU == 1
	// Morris Lee
	cuda_procImg(g_can, g_ang, g_nor, image1, initial);
	smplHoG64(sHoG, g_ang, g_nor);
#endif
}

void defcan2(double* g_can, unsigned char* image1)
{
	/* definite canonicalization */
	int x, y;
	double mean, norm, ratio; // mean: mean value, norm: normal factor, ratio:
	int margine = CANMARGIN / 2;
	int npo; // number of point

	// npo = (ROW - 2 * MARGINE) * (COL - 2 * MARGINE);
	npo = 0;
	mean = norm = 0.0;
	for (y = margine; y < ROW - margine; y++)
	{
		for (x = margine; x < COL - margine; x++)
		{
			if (image1[y*COL+x] != WHITE)
			{
				mean += (double)image1[y*COL+x];
				norm += (double)image1[y*COL+x] * (double)image1[y*COL+x];
				npo++;
			}
		}
	}
	mean /= (double)npo;
	norm -= (double)npo * mean * mean;
	if (norm == 0.0)
		norm = 1.0;
	ratio = 1.0 / sqrt(norm);
	for (y = margine; y < ROW - margine; y++)
	{
		for (x = margine; x < COL - margine; x++)
		{
			if (image1[y*COL+x] != WHITE)
			{
				g_can[y*COL+x] = ratio * ((double)image1[y*COL+x] - mean);
			}
			else
			{
				g_can[y*COL+x] = 0.0;
			}
		}
	}
}

void roberts8(int* g_ang, double* g_nor, unsigned char* image1)
{
	/* extraction of gradient information by Roberts operator */
	/* with 8-directional codes and strength */
	double delta_RD, delta_LD;
	double angle;
	int x, y; /* Loop variable */

	/* angle & norm of gradient vector calculated
	 by Roberts operator */
	for (y = 0; y < ROW; y++)
	{
		for (x = 0; x < COL; x++)
		{
			g_ang[y*COL+x] = -1;
			g_nor[y*COL+x] = 0.0;
		}
	}

	for (y = 0; y < ROW - 1; y++)
	{
		for (x = 0; x < COL - 1; x++)
		{
			//printf("(%d, %d) ang = %d,  norm = %f\n", x, y, g_ang[y*COL + x], g_nor[y*COL + x]);
			delta_RD = image1[y*COL+x + 1] - image1[(y + 1)*COL+x];
			delta_LD = image1[y*COL+x] - image1[(y + 1)*COL+x + 1];
			g_nor[y*COL+x] = sqrt(delta_RD * delta_RD + delta_LD * delta_LD);

			if (g_nor[y*COL+x] == 0.0 || delta_RD * delta_RD + delta_LD * delta_LD < NoDIRECTION * NoDIRECTION)
				continue;

			if (abs(delta_RD) == 0.0)
			{
				if (delta_LD > 0)
					g_ang[y*COL+x] = 3;
				if (delta_LD < 0)
					g_ang[y*COL+x] = 7;
			}
			else
			{
				angle = atan2(delta_LD, delta_RD);
				if (angle > 7.0 / 8.0 * PI)
					g_ang[y*COL+x] = 5;
				else if (angle > 5.0 / 8.0 * PI)
					g_ang[y*COL+x] = 4;
				else if (angle > 3.0 / 8.0 * PI)
					g_ang[y*COL+x] = 3;
				else if (angle > 1.0 / 8.0 * PI)
					g_ang[y*COL+x] = 2;
				else if (angle > -1.0 / 8.0 * PI)
					g_ang[y*COL+x] = 1;
				else if (angle > -3.0 / 8.0 * PI)
					g_ang[y*COL+x] = 0;
				else if (angle > -5.0 / 8.0 * PI)
					g_ang[y*COL+x] = 7;
				else if (angle > -7.0 / 8.0 * PI)
					g_ang[y*COL+x] = 6;
				else
					g_ang[y*COL+x] = 5;
			}
		}
	}
}

void smplHoG64(char* sHoG, int* g_ang, double* g_nor)
{
	int x, y, dx, dy, dir;
	double HoG[8];
	int HoGIdx[8];

	for (y = 0; y < ROW - 4; y++)
	{
		for (x = 0; x < COL - 4; x++)
		{
			sHoG[y*(COL-4)+x] = -1;
			// initialize
			for (dir = 0; dir < 8; dir++)
			{
				HoG[dir] = 0.0;
				HoGIdx[dir] = dir + 1;
			}
			// calculate HoG
			for (dy = y; dy < y + 5; dy++)
			{
				for (dx = x; dx < x + 5; dx++)
				{
					if (g_ang[dy*COL+dx] == -1)
						break;
					HoG[g_ang[dy*COL+dx]] += g_nor[dy*COL+dx];
				}
			}
			// sort of the 8 HoG (One step of bubble sort)
			for (dir = 7; dir > 0; dir--)
			{
				if (HoG[dir] >= HoG[dir - 1])
				{
					changeValue<double>(&HoG[dir], &HoG[dir - 1]);
					changeValue<int>(&HoGIdx[dir], &HoGIdx[dir - 1]);
				}
			}
			for (dir = 7; dir > 1; dir--)
			{
				if (HoG[dir] >= HoG[dir - 1])
				{
					changeValue<double>(&HoG[dir], &HoG[dir - 1]);
					changeValue<int>(&HoGIdx[dir], &HoGIdx[dir - 1]);
				}
			}
			// calculate the direction
			if (HoG[0] > SHoGTHRE)
			{
				sHoG[y*(COL-4)+x] = (char)HoGIdx[0];
				if (HoG[1] > SHoGSECONDTHRE * HoG[0])
				{
					sHoG[y*(COL-4)+x] = sHoG[y*(COL-4)+x] * 10 + (char)HoGIdx[1];
				}
			}
			// printf("<%d, %d> HoG = %d \n", y, x, sHoG[y][x]);
		}
	}
}

void bilinear_normal_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
	unsigned char* image1, unsigned char* image2,int initial = 1)
{
	/* projection transformation of the image by bilinear interpolation */
	double inv_gpt[3][3];
	inverse3x3(gpt, inv_gpt);
	if (isGPU == 0)
		bilinear_normal_inverse_projection(inv_gpt, x_size1, y_size1, x_size2, y_size2, image1, image2);
	else
		cuda_bilinear_normal_inverse_projection(inv_gpt, x_size1, y_size1, x_size2, y_size2, image1, image2, initial);
}

void bilinear_normal_inverse_projection(double gpt[3][3], int x_size1, int y_size1, int x_size2, int y_size2,
	unsigned char* image1, unsigned char* image2)
{
	/* inverse projection transformation of the image by bilinear interpolation */
	int x, y;
	double inVect[3], outVect[3];
	double x_new, y_new, x_frac, y_frac;
	double gray_new;
	int m, n;
	int cx, cy, cx2, cy2;
	int idx;

	/* output image generation by bilinear interpolation */
	// x_size2 = x_size1;
	// y_size2 = y_size1;
	if (y_size1 == ROW)
	{
		cx = CX, cy = CY;
		cx2 = CX2, cy2 = CY2;
	}
	else
	{
		cx = CX2, cy = CY2;
		cx2 = CX, cy2 = CY;
	}
	inVect[2] = 1.0;
	for (y = 0; y < y_size1; y++)
	{
		inVect[1] = y - cy;
		for (x = 0; x < x_size1; x++)
		{
			inVect[0] = x - cx;
			multiplyVect3x3(gpt, inVect, outVect);
			x_new = outVect[0] / outVect[2] + cx2;
			y_new = outVect[1] / outVect[2] + cy2;
			m = (int)floor(x_new);
			n = (int)floor(y_new);
			x_frac = x_new - m;
			y_frac = y_new - n;

			if (m >= 0 && m + 1 < x_size2 && n >= 0 && n + 1 < y_size2)
			{
				gray_new = (1.0 - y_frac) * ((1.0 - x_frac) * image1[n*x_size2+m] + x_frac * image1[n*x_size2+m + 1]) + y_frac * ((1.0 - x_frac) * image1[(n + 1)*x_size2+m] + x_frac * image1[(n + 1)*x_size2+m + 1]);
				image2[y*x_size1+x] = (unsigned char)gray_new;
			}
			else
			{
#ifdef BACKGBLACK
				image2[y*x_size1+x] = BLACK;
#else
				image2[y*x_size1+x] = WHITE;
#endif
			}
			//printf("(%d %d): inVect = (%f; %f; %f), outVect = (%f; %f; %f) \n", y, x, inVect[0], inVect[1], inVect[2], outVect[0], outVect[1], outVect[2]);
			//printf("%d %d = %d %d || n = %d, m = %d, x_new = %f, y_new = %f, x_frac = %f, y_frac = %f \n", y, x, image2[y][x], image1[y][x], n, m, x_new, y_new, x_frac, y_frac);
		}
	}
}