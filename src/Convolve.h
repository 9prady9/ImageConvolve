#ifndef __CONVOLVE_H__
#define __CONVOLVE_H__

#include <stdio.h>

using uchar = unsigned char;

struct Kernel
{
	int kr;
	float* krValues;
	Kernel();
	void setKernelRadius(int fKernelRadius);
	void setCellValue(int fRow, int fCol, float fValue);
};

class Convolve
{
public:
	Convolve();
	void setKernelData(const Kernel &fKernel);
	void setImageData(const uchar* fImageData, int fImageWidth, int fImageHeight);
	uchar* getConvolvedImage();
	float cudaConvolve();
	~Convolve();

	/* Public Atrribute to enable saving of output outside the class */
	uchar*	host_ConvolvedImage;

private:
	/* Attributes */
	bool	mIsKernelSet;
	bool	mIsImageSet;
	int		mImageWidth;
	int		mImageHeight;
	int		mKernelSize;
};

#endif //CONVOLVE_CUDA_H
