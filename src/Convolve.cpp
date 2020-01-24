#include "Convolve.h"

#include "conv.cuh"
#include <qimage.h>
#include <qdebug.h>

#include <chrono>
#include <iostream>
#include <stdlib.h>

//std::chrono::time_point<std::chrono::system_clock>
using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

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
	double time = 0.0;

	if(mIsKernelSet && mIsImageSet) {
        auto start = std::chrono::high_resolution_clock::now();

		convolve(mKernelSize);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;

        time = diff.count();

		std::cout<<"computation time on device: "<< time << std::endl;

		/* Copy back convolved image to host and show it */
		memCpyImageDeviceToHost(host_ConvolvedImage);
		QImage output(host_ConvolvedImage,mImageWidth,mImageHeight, QImage::Format_ARGB32);
	} else {
		if(!mIsKernelSet)
			printf("Prerequisite: Kernel not setup. \n");
		if(!mIsImageSet)
			printf("Prerequisite: Image not setup. \n");
		time = 0.0;
	}
	return time;
}

Convolve::~Convolve()
{
	free(host_ConvolvedImage);
}
