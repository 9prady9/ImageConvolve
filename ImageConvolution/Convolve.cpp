#include "Convolve.h"

#include "conv.cuh"
#include <malloc.h>
#include <qimage.h>
#include <qdebug.h>
#include <iostream>
#include <chrono>

//std::chrono::time_point<std::chrono::system_clock>
typedef std::chrono::time_point<std::chrono::system_clock> ChronoTimer;

Kernel::Kernel()
{
	krValues = 0;
	kr = -1;
}

void Kernel::setKernelRadius(int fKernelRadius)
{
	if(krValues) free(krValues);
	kr = fKernelRadius;
	int temp = 2*fKernelRadius+1;
	krValues = (float*)malloc(temp*temp*sizeof(float));
}

void Kernel::setCellValue(int fRow, int fCol, float fValue)
{
	int temp = 2*kr+1;
	krValues[fRow*temp+fCol] = fValue;
}

Convolve::Convolve()
{
	mIsKernelSet		= false;
	mIsImageSet			= false;
	host_ConvolvedImage = 0;
	initMemObject();
	QueryPerformanceFrequency(&mFrequency);
}

void Convolve::setKernelData(const Kernel &fKernel)
{
	int temp			= fKernel.kr*2 + 1;
	setKernelOnDevice(fKernel.krValues,temp*temp);
	mKernelSize			= fKernel.kr;
	mIsKernelSet		= true;
}

void Convolve::setImageData(const uchar* fImageData, int fImageWidth, int fImageHeight)
{	
	free(host_ConvolvedImage);

	setImageOnDevice(fImageData, fImageWidth, fImageHeight);
	mIsImageSet = true;
	
	mImageWidth		= fImageWidth;
	mImageHeight	= fImageHeight;		
	host_ConvolvedImage = (uchar*)malloc(mImageWidth*mImageHeight*4*sizeof(uchar));
}

uchar* Convolve::getConvolvedImage()
{
	return host_ConvolvedImage;
}

float Convolve::cudaConvolve()
{
	float time;
	if(mIsKernelSet && mIsImageSet)
	{
		LARGE_INTEGER start, end;
		
		QueryPerformanceCounter(&start);
		convolve(mKernelSize);
		QueryPerformanceCounter(&end);

		time = static_cast<double>(end.QuadPart- start.QuadPart) / mFrequency.QuadPart;
		time *= 1000.0f;

		std::cout<<"computation time on device: "<<time<<" ms\n";

		/* Copy back convolved image to host and show it */		
		memCpyImageDeviceToHost(host_ConvolvedImage);
		QImage output(host_ConvolvedImage,mImageWidth,mImageHeight, QImage::Format_ARGB32);
	} else {
		if(!mIsKernelSet)
			printf("Prerequisite: Kernel not setup. \n");
		if(!mIsImageSet)
			printf("Prerequisite: Image not setup. \n");
		time = 0.0f;
	}
	return time;
}

Convolve::~Convolve()
{
	free(host_ConvolvedImage);
}