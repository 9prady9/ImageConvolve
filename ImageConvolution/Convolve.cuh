#ifndef CONVOLVE_CUDA_H
#define CONVOLVE_CUDA_H

#include <stdio.h>
#include <curand_kernel.h>

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
typedef unsigned char uchar;

int ceil(int numer, int denom);

struct Kernel
{
	int kr;
	int* krValues;
	Kernel();
	void setKernelRadius(int fKernelRadius);
	void setCellValue(int fRow, int fCol, int fValue);
};

__global__ void convolveKernel(uchar* fCanvas, int fCanvasWidth, int fCanvasHeight, uchar* dev_fConvolvedImage, int fImgWidth, int fImgHeight, int fKernelSize, dim3 fMyLoad );

class Convolve
{
public:
	Convolve();
	void setKernelData(const Kernel &fKernel);
	void setImageData(const uchar* fImageData, int fImageWidth, int fImageHeight);
	uchar* getConvolvedImage();
	void initCUDA();
	void cudaConvolve();
	void destroyCUDA();
	~Convolve();
	/* Public Atrribute to enable saving of output outside the class */
	uchar*	host_ConvolvedImage;

private:
	bool	mIsKernelSet;
	bool	mIsImageSet;
	bool	mIsCUDAInit;
	bool	mIsComputeDone;
	dim3	mThreadsPerBlock;
	dim3	mPerThreadLoad;
	dim3	mGrid;
	uchar*	host_ARGB_channel;
	int		mImageWidth;
	int		mImageHeight;
	int		mCanvasWidth;
	int		mCanvasHeight;

	cudaEvent_t		mStart, mStop;
};

#endif //CONVOLVE_CUDA_H