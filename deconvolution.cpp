/*
============================================================================
Name : deconv.c
Author : Tinniam V Ganesh
Version :
Copyright :
Description : Deconvolution in OpenCV
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"

int main(int argc, char ** argv)
{
int height,width,step,channels;
uchar* data;
uchar* data1;
int i,j,k;
float s;

CvMat *dft_A;
CvMat *dft_B;
CvMat *dft_C;
IplImage* im;
IplImage* im1;

IplImage* image_ReB;
IplImage* image_ImB;

IplImage* image_ReC;
IplImage* image_ImC;
IplImage* complex_ImC;

IplImage* image_ReDen;
IplImage* image_ImDen;

FILE *fp;
fp = fopen("test.txt","w+");

int dft_M,dft_N;
int dft_M1,dft_N1;
CvMat* cvShowDFT();
void cvShowInvDFT();

im1 = cvLoadImage( "kutty-1.jpg",1 );
cvNamedWindow("original-color", 0);
cvShowImage("original-color", im1);
im = cvLoadImage( "kutty-1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
if( !im )
return -1;

cvNamedWindow("original-gray", 0);
cvShowImage("original-gray", im);
// Create blur kernel (non-blind)
//float vals[]={.000625,.000625,.000625,.003125,.003125,.003125,.000625,.000625,.000625};
//float vals[]={-0.167,0.333,0.167,-0.167,.333,.167,-0.167,.333,.167};

float vals[]={.055,.055,.055,.222,.222,.222,.055,.055,.055};
CvMat kernel = cvMat(3, // number of rows
3, // number of columns
CV_32FC1, // matrix data type
vals);
IplImage* k_image_hdr;
IplImage* k_image;

k_image_hdr = cvCreateImageHeader(cvSize(3,3),IPL_DEPTH_64F,2);
k_image = cvCreateImage(cvSize(3,3),IPL_DEPTH_64F,1);
k_image = cvGetImage(&kernel,k_image_hdr);

/*IplImage* k_image;
k_image = cvLoadImage( "kernel4.bmp",0 );*/
cvNamedWindow("blur kernel", 0);

height = k_image->height;
width = k_image->width;
step = k_image->widthStep;

channels = k_image->nChannels;
//data1 = (float *)(k_image->imageData);
data1 = (uchar *)(k_image->imageData);

cvShowImage("blur kernel", k_image);

dft_M = cvGetOptimalDFTSize( im->height - 1 );
dft_N = cvGetOptimalDFTSize( im->width - 1 );

//dft_M1 = cvGetOptimalDFTSize( im->height+99 - 1 );
//dft_N1 = cvGetOptimalDFTSize( im->width+99 - 1 );

dft_M1 = cvGetOptimalDFTSize( im->height+3 - 1 );
dft_N1 = cvGetOptimalDFTSize( im->width+3 - 1 );

// Perform DFT of original image
dft_A = cvShowDFT(im, dft_M1, dft_N1,"original");
//Perform inverse (check & comment out) - Commented as it overwrites dft_A
//cvShowInvDFT(im,dft_A,dft_M1,dft_N1,fp, "original");

// Perform DFT of kernel
dft_B = cvShowDFT(k_image,dft_M1,dft_N1,"kernel");
//Perform inverse of kernel (check & comment out) - commented as it overwrites dft_B
//cvShowInvDFT(k_image,dft_B,dft_M1,dft_N1,fp, "kernel");

// Multiply numerator with complex conjugate
dft_C = cvCreateMat( dft_M1, dft_N1, CV_64FC2 );

printf("%d %d %d %d\n",dft_M,dft_N,dft_M1,dft_N1);

// Multiply DFT(blurred image) * complex conjugate of blur kernel
cvMulSpectrums(dft_A,dft_B,dft_C,CV_DXT_MUL_CONJ);

// Split Fourier in real and imaginary parts
image_ReC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
complex_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 2);

printf("%d %d %d %d\n",dft_M,dft_N,dft_M1,dft_N1);

//cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );
cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );

// Compute A^2 + B^2 of denominator or blur kernel
image_ReB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);

// Split Real and imaginary parts
cvSplit( dft_B, image_ReB, image_ImB, 0, 0 );
cvPow( image_ReB, image_ReB, 2.0);
cvPow( image_ImB, image_ImB, 2.0);
cvAdd(image_ReB, image_ImB, image_ReB,0);

//Divide Numerator/A^2 + B^2
cvDiv(image_ReC, image_ReB, image_ReC, 1.0);
cvDiv(image_ImC, image_ReB, image_ImC, 1.0);

// Merge Real and complex parts
cvMerge(image_ReC, image_ImC, NULL, NULL, complex_ImC);

// Perform Inverse
cvShowInvDFT(im, complex_ImC,dft_M1,dft_N1,fp,"deblur");
cvWaitKey(-1);
return 0;
}

CvMat* cvShowDFT(im, dft_M, dft_N,src)
IplImage* im;
int dft_M, dft_N;
char* src;
{
IplImage* realInput;
IplImage* imaginaryInput;
IplImage* complexInput;

CvMat* dft_A, tmp;

IplImage* image_Re;
IplImage* image_Im;

char str[80];

double m, M;

realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);

cvScale(im, realInput, 1.0, 0.0);
cvZero(imaginaryInput);
cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

// copy A to dft_A and pad dft_A with zeros
cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
cvCopy( complexInput, &tmp, NULL );
if( dft_A->cols > im->width )
{
cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols - im->width, im->height));
cvZero( &tmp );
}

// no need to pad bottom part of dft_A with zeros because of
// use nonzero_rows parameter in cvDFT() call below

cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );

strcpy(str,"DFT -");
strcat(str,src);
cvNamedWindow(str, 0);

// Split Fourier in real and imaginary parts
cvSplit( dft_A, image_Re, image_Im, 0, 0 );

// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
cvPow( image_Re, image_Re, 2.0);
cvPow( image_Im, image_Im, 2.0);
cvAdd( image_Re, image_Im, image_Re, NULL);
cvPow( image_Re, image_Re, 0.5 );

// Compute log(1 + Mag)
cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
cvLog( image_Re, image_Re ); // log(1 + Mag)

cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
cvShowImage(str, image_Re);
return(dft_A);
}

void cvShowInvDFT(im, dft_A, dft_M, dft_N,fp, src)
IplImage* im;
CvMat* dft_A;
int dft_M,dft_N;
FILE *fp;
char* src;
{

IplImage* realInput;
IplImage* imaginaryInput;
IplImage* complexInput;

IplImage * image_Re;
IplImage * image_Im;

double m, M;
char str[80];

realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);

image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

//cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, complexInput->height );
cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, dft_M);
strcpy(str,"DFT INVERSE - ");
strcat(str,src);
cvNamedWindow(str, 0);

// Split Fourier in real and imaginary parts
cvSplit( dft_A, image_Re, image_Im, 0, 0 );

// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
cvPow( image_Re, image_Re, 2.0);
cvPow( image_Im, image_Im, 2.0);
cvAdd( image_Re, image_Im, image_Re, NULL);
cvPow( image_Re, image_Re, 0.5 );

// Compute log(1 + Mag)
cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
cvLog( image_Re, image_Re ); // log(1 + Mag)

cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
//cvCvtColor(image_Re, image_Re, CV_GRAY2RGBA);

cvShowImage(str, image_Re);
}

 
