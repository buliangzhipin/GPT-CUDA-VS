#pragma once

#define isGPU 1


#define DATATYPE 2
#define IMGDIR "."
#define MAXITER 100 // Maximum iteration times
#define MAXNR 5     // Maximum Newton-Raphson iterations

/* --------------Fixed parameters-------------- */
#define BLACK 0            // color of black
#define WHITE 255          // color of white
#define NoDIRECTION 20.0   // The threshold of non gradient direction if the norm of gradient is weak
#define NOHoG 8            // The threshold of non HoG feature
#define SHoGTHRE 300.0     // The first direction should over this value
#define SHoGSECONDTHRE 0.5 // The threshold of the second direction of the simplified HoG pattern
#define DNNSWITCHTHRE 2.0  // The threshold of switch the method of dnn calculation
// #define TRUNC			1500
#define TRUNC (2 * ROW - 1) * (2 * COL - 1) / 1
#define PI 3.141592654     // value of pi
#define EPS 0.000001       // like zero
#define EPS2 0.000001      // like zero2
#define WGT 1.5            /* Gauss window size weight */
#define MAX_IMAGESIZE 1024 //
#define MAX_BRIGHTNESS 255 /* Maximum gray level */
#define GRAYLEVEL 256      /* No. of gray levels */
#define MAX_FILENAME 256   /* Filename length limit */
#define MAX_BUFFERSIZE 256
#define HoGTHRESHOLD 8
#define HoGTHRESHOLD3 3
#define TRUE 1
#define FALSE 0
/* --------------Fixed parameters-------------- */

/* initial conditions */
#define NONELEMENT /* use non-elemental matrix as initial condition */
#define ZOOM 1.7   /* Zoom rate for initial matrix */
#define BETA 1.0   /* Relation between alpha and beta */
#define ROT 0.0    /* Rotation angle for initial matrix */
#define B1 0.0     /*  */
#define B2 0.0     /*  */

#if DATATYPE == 2 /* 20 300 2.0 */
#define COL 170   /* Horizontal size of image  */
#define ROW 136   /* Vertical   size of image  */
#define COL2 340  /* Horizontal size of image  */
#define ROW2 272  /* Vertical   size of image  */
#define CX 85
#define CY 68
#define CX2 170
#define CY2 136

#define MARGINE 0   /* Margine size              */
#define CANMARGIN 0 /* Margine size for calculate crr */
#define TsIMAGE "sample_boat/img2_small2"
#define RgIMAGE "sample_boat/img1_small"
#define CENTERCORRELATION
/* acc: 3.5,
 * Memo of the parameter set
 * img1_small to img2_small2 --> ZOOM = 1.7;
 * img1_small to img3_small2 --> ZOOM = 2.0, ROT = 45.0;
 * img1_small to img4_small2 --> ZOOM = 1.4, ROT = 90.0;
 * img1_small to img5_small2 --> ZOOM = 1.1;
 * img1_small to img6_small2 --> ZOOM = 0.8, ROT = 45.0; (acc ZOOM = 0.8)
 */
#endif